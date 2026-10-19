"""Holdout comparison for the vocal-melody MLP on OpenScore Lieder.

The question is whether syllable and phrase features improve next-interval
and duration prediction over a key-relative interval model. Every row is
scored on the same songs, in the same scale-degree pool the trainer uses.

Two protocols:

- ``teacher_forced``: each step conditions on the true previous degrees.
  This measures next-note prediction.
- ``rollout``: each step conditions on the model's own previous degrees, which
  is how Generate vocal runs. Interval error here is the step the model took
  versus the step the score took, matching ``evaluate_songs``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calliope.audio.lieder_melody import (
    INTERVAL_CLIP,
    ParsedSong,
    duration_log,
    iter_mscx,
    iter_parsed_songs,
    log_to_beats,
    nearest_pool_index,
)
from calliope.audio.vocal_melody_ml import (
    _forward,
    fit_mlp_weights,
    holdout_indices,
    syllable_features,
)
from calliope.audio.vocal_synth import (
    VOICE_RANGES,
    _NOTE_TO_MIDI,
    _SCALES,
    _scale_pitches,
)

# Column order of syllable_features.
POS = 0
PHRASE_INDEX = 1
PHRASE_END = 2
PHRASE_START = 3
VOWEL = 4
TOKEN_LENGTH = 5
GENRE = 6
ARRANGEMENT = 7
PREV_DEGREE = 8
PREV2_DEGREE = 9
VOICE_CENTER = 10
LATER_SECTION = 11

LYRIC_COLUMNS = (VOWEL, TOKEN_LENGTH)
PHRASE_COLUMNS = (PHRASE_INDEX, PHRASE_END, PHRASE_START, LATER_SECTION)
PITCH_COLUMNS = (PREV_DEGREE, PREV2_DEGREE)

HOLDOUT_FRACTION = 0.2
SPLIT_SEED = 7
BOOTSTRAP_SEED = 11
N_BOOTSTRAP = 1000
GENRE_PRESET = "acoustic"
ARRANGEMENT_STYLE = "through-composed"
REPORT_PATH = Path(__file__).resolve().parents[4] / "docs" / "melody-holdout.json"


@dataclass(frozen=True)
class Step:
    tok: str
    i: int
    n: int
    phrase_idx: int
    n_phrases: int
    phrase_end: bool
    phrase_start: bool
    true_idx: int
    beats: float
    phrase_pos: float
    pool_len: int
    lo: int
    hi: int


@dataclass
class SongScore:
    interval_abs: float = 0.0
    duration_abs: float = 0.0
    contour_hits: int = 0
    degree_hits: int = 0
    n: int = 0

    @property
    def interval_mae(self) -> float:
        return self.interval_abs / self.n if self.n else 0.0


def steps_for_song(song: ParsedSong) -> list[Step]:
    """One row per lyric note, using the same pool and phrase counts as training."""
    notes = list(song.notes[:80])
    if len(notes) < 6:
        return []
    from calliope.audio.lieder_melody import voice_for_midis

    voice = voice_for_midis([note.midi for note in notes])
    lo, hi = VOICE_RANGES.get(voice, VOICE_RANGES["soprano"])
    root = _NOTE_TO_MIDI.get(song.root, 67)
    intervals = _SCALES.get(song.scale, _SCALES["major"])
    pool = _scale_pitches(root, intervals, lo, hi)
    n_phrases = 1 + sum(1 for note in notes[:-1] if note.phrase_end)
    phrase_pos = _phrase_positions(notes)
    steps: list[Step] = []
    phrase_idx = 0
    n = len(notes)
    for i, note in enumerate(notes):
        steps.append(
            Step(
                tok=note.tok,
                i=i,
                n=n,
                phrase_idx=phrase_idx,
                n_phrases=max(1, n_phrases),
                phrase_end=note.phrase_end,
                phrase_start=i == 0 or notes[i - 1].phrase_end,
                true_idx=nearest_pool_index(note.midi, pool),
                beats=float(note.beats),
                phrase_pos=phrase_pos[i],
                pool_len=len(pool),
                lo=lo,
                hi=hi,
            )
        )
        if note.phrase_end:
            phrase_idx += 1
    return steps


def _phrase_positions(notes: list) -> list[float]:
    groups: list[tuple[int, int]] = []
    start = 0
    for i, note in enumerate(notes):
        if note.phrase_end or i == len(notes) - 1:
            groups.append((start, i))
            start = i + 1
    pos = [0.0] * len(notes)
    for a, b in groups:
        span = max(1, b - a)
        for j in range(a, b + 1):
            pos[j] = (j - a) / span
    return pos


def make_feature(step: Step, prev: float, prev2: float) -> np.ndarray:
    return syllable_features(
        tok=step.tok,
        i=step.i,
        n=step.n,
        phrase_idx=step.phrase_idx,
        n_phrases=step.n_phrases,
        phrase_end=step.phrase_end,
        phrase_start=step.phrase_start,
        genre_preset=GENRE_PRESET,
        arrangement_style=ARRANGEMENT_STYLE,
        prev_idx=prev,
        prev2_idx=prev2,
        pool_len=step.pool_len,
        lo=step.lo,
        hi=step.hi,
    )


def apply_drop(feat: np.ndarray, drop: tuple[int, ...]) -> np.ndarray:
    if not drop:
        return feat
    out = np.array(feat, dtype=np.float32, copy=True)
    out[list(drop)] = 0.0
    return out


def matrix_from_steps(
    steps: list[Step],
    drop: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Teacher-forced (X, y) rows. y is [clipped interval, duration log]."""
    if not steps:
        return np.zeros((0, 12), np.float32), np.zeros((0, 2), np.float32)
    prev = float(steps[0].pool_len // 2)
    prev2 = prev
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for step in steps:
        feat = apply_drop(make_feature(step, prev, prev2), drop)
        delta = float(np.clip(step.true_idx - prev, -INTERVAL_CLIP, INTERVAL_CLIP))
        xs.append(feat)
        ys.append(np.array([delta, duration_log(step.beats)], dtype=np.float32))
        prev2 = prev
        prev = float(step.true_idx)
    return np.stack(xs), np.stack(ys)


def _clip_idx(idx: int, pool_len: int) -> int:
    return int(np.clip(idx, 0, pool_len - 1))


def _true_delta(true_idx: int, prev_true: int) -> int:
    return int(np.clip(true_idx - prev_true, -INTERVAL_CLIP, INTERVAL_CLIP))


def duration_bin(step: Step) -> str:
    """Where the syllable sits in its phrase. Phrase-final wins over phrase-initial."""
    if step.phrase_end:
        return "end"
    if step.phrase_start:
        return "start"
    if step.phrase_pos < 0.5:
        return "early"
    return "late"


def _median_int(values: list[int], fallback: int = 0) -> int:
    if not values:
        return fallback
    return int(np.round(float(np.median(values))))


def _mode_int(values: list[int], fallback: int = 0) -> int:
    if not values:
        return fallback
    counts: dict[int, int] = defaultdict(int)
    for v in values:
        counts[v] += 1
    best = max(counts, key=lambda k: (counts[k], -abs(k), k))
    return int(best)


def fit_interval_tables(
    songs: list[list[Step]],
) -> dict[str, object]:
    """Constant, unigram, and first-order interval tables, plus duration bins."""
    deltas: list[int] = []
    by_prev: dict[int | None, list[int]] = defaultdict(list)
    by_bin: dict[str, list[float]] = defaultdict(list)
    all_beats: list[float] = []
    for steps in songs:
        if not steps:
            continue
        prev = steps[0].pool_len // 2
        prev_key: int | None = None
        for step in steps:
            delta = _true_delta(step.true_idx, prev)
            deltas.append(delta)
            by_prev[prev_key].append(delta)
            beats = float(np.clip(step.beats, 0.25, 4.0))
            by_bin[duration_bin(step)].append(beats)
            all_beats.append(beats)
            prev_key = delta
            prev = step.true_idx
    global_median = _median_int(deltas, 0)
    markov = {key: _median_int(vals, global_median) for key, vals in by_prev.items()}
    beat_median = float(np.median(all_beats)) if all_beats else 0.5
    bins = {key: float(np.median(vals)) for key, vals in by_bin.items()}
    return {
        "median_interval": global_median,
        "mode_interval": _mode_int(deltas, global_median),
        "markov": markov,
        "median_beats": beat_median,
        "beats_by_bin": bins,
    }


def _beats_for(step: Step, tables: dict[str, object], *, positional: bool) -> float:
    if not positional:
        return float(tables["median_beats"])
    bins = tables["beats_by_bin"]
    assert isinstance(bins, dict)
    return float(bins.get(duration_bin(step), tables["median_beats"]))


def _markov_delta(tables: dict[str, object], prev_key: int | None) -> int:
    table = tables["markov"]
    assert isinstance(table, dict)
    if prev_key in table:
        return int(table[prev_key])
    return int(tables["median_interval"])


def predict_constant(
    steps: list[Step],
    delta: int,
    beats_of,
    *,
    teacher: bool,
) -> tuple[list[int], list[float]]:
    return _predict_interval(steps, lambda _prev: delta, beats_of, teacher=teacher)


def predict_markov(
    steps: list[Step],
    tables: dict[str, object],
    beats_of,
    *,
    teacher: bool,
) -> tuple[list[int], list[float]]:
    def choose(prev_key: int | None) -> int:
        return _markov_delta(tables, prev_key)

    return _predict_interval(steps, choose, beats_of, teacher=teacher)


def _predict_interval(
    steps: list[Step],
    choose,
    beats_of,
    *,
    teacher: bool,
) -> tuple[list[int], list[float]]:
    if not steps:
        return [], []
    prev_true = steps[0].pool_len // 2
    prev_pred = prev_true
    prev_key: int | None = None
    idxs: list[int] = []
    beats: list[float] = []
    for step in steps:
        delta = int(choose(prev_key))
        origin = prev_true if teacher else prev_pred
        new_idx = _clip_idx(origin + delta, step.pool_len)
        idxs.append(new_idx)
        beats.append(float(beats_of(step)))
        if teacher:
            prev_key = _true_delta(step.true_idx, prev_true)
            prev_true = step.true_idx
        else:
            prev_key = int(np.clip(new_idx - origin, -INTERVAL_CLIP, INTERVAL_CLIP))
            prev_pred = new_idx
    return idxs, beats


def predict_mlp(
    steps: list[Step],
    weights: dict[str, np.ndarray],
    drop: tuple[int, ...],
    *,
    teacher: bool,
) -> tuple[list[int], list[float]]:
    if not steps:
        return [], []
    prev = float(steps[0].pool_len // 2)
    prev2 = prev
    pred_idx = int(prev)
    idxs: list[int] = []
    beats: list[float] = []
    for step in steps:
        feat = apply_drop(make_feature(step, prev, prev2), drop)
        delta, log_dur = _forward(feat, weights)
        origin = int(prev) if teacher else pred_idx
        new_idx = _clip_idx(origin + int(round(float(delta))), step.pool_len)
        idxs.append(new_idx)
        beats.append(log_to_beats(float(log_dur)))
        if teacher:
            prev2 = prev
            prev = float(step.true_idx)
        else:
            prev2 = float(pred_idx)
            pred_idx = new_idx
            prev = float(pred_idx)
    return idxs, beats


def score_song(
    steps: list[Step],
    pred_idxs: list[int],
    pred_beats: list[float],
    *,
    teacher: bool,
) -> SongScore:
    """Interval error uses the training clip of ±5 on the reference step."""
    score = SongScore()
    if not steps:
        return score
    prev_true = steps[0].pool_len // 2
    prev_pred = prev_true
    for step, pred_idx, pred_beat in zip(steps, pred_idxs, pred_beats, strict=True):
        t_delta = _true_delta(step.true_idx, prev_true)
        origin = prev_true if teacher else prev_pred
        p_delta = int(pred_idx) - origin
        ref_beats = float(np.clip(step.beats, 0.25, 4.0))
        score.interval_abs += abs(p_delta - t_delta)
        score.duration_abs += abs(float(pred_beat) - ref_beats)
        score.contour_hits += int(np.sign(p_delta) == np.sign(t_delta))
        degree_origin = prev_true if teacher else prev_pred
        reached = _clip_idx(degree_origin + p_delta, step.pool_len)
        score.degree_hits += int(reached == step.true_idx)
        prev_true = step.true_idx
        if not teacher:
            prev_pred = int(pred_idx)
        score.n += 1
    return score


def _summarize(parts: list[SongScore], rng: np.random.Generator) -> dict[str, float | list[float] | int]:
    n = sum(p.n for p in parts)
    if not n:
        return {
            "interval_mae": -1.0,
            "duration_mae": -1.0,
            "contour_acc": -1.0,
            "degree_acc": -1.0,
            "interval_mae_macro": -1.0,
            "interval_mae_ci95": [-1.0, -1.0],
            "duration_mae_ci95": [-1.0, -1.0],
            "contour_acc_ci95": [-1.0, -1.0],
            "n_notes": 0,
            "n_songs": 0,
        }
    interval = sum(p.interval_abs for p in parts) / n
    duration = sum(p.duration_abs for p in parts) / n
    contour = sum(p.contour_hits for p in parts) / n
    degree = sum(p.degree_hits for p in parts) / n
    macro = float(np.mean([p.interval_mae for p in parts if p.n]))

    def ci(total) -> list[float]:
        boots: list[float] = []
        n_songs = len(parts)
        for _ in range(N_BOOTSTRAP):
            draw = rng.integers(0, n_songs, size=n_songs)
            num = 0.0
            den = 0
            for i in draw:
                num += total(parts[int(i)])
                den += parts[int(i)].n
            boots.append(num / den if den else 0.0)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return [float(lo), float(hi)]

    return {
        "interval_mae": float(interval),
        "duration_mae": float(duration),
        "contour_acc": float(contour),
        "degree_acc": float(degree),
        "interval_mae_macro": macro,
        "interval_mae_ci95": ci(lambda p: p.interval_abs),
        "duration_mae_ci95": ci(lambda p: p.duration_abs),
        "contour_acc_ci95": ci(lambda p: p.contour_hits),
        "n_notes": int(n),
        "n_songs": int(len(parts)),
    }


def _eval_pair(hold: list[list[Step]], predict) -> dict[str, dict]:
    forced_parts: list[SongScore] = []
    roll_parts: list[SongScore] = []
    for steps in hold:
        f_idx, f_beats = predict(steps, teacher=True)
        r_idx, r_beats = predict(steps, teacher=False)
        forced_parts.append(score_song(steps, f_idx, f_beats, teacher=True))
        roll_parts.append(score_song(steps, r_idx, r_beats, teacher=False))
    return {
        "teacher_forced": _summarize(forced_parts, np.random.default_rng(BOOTSTRAP_SEED)),
        "rollout": _summarize(roll_parts, np.random.default_rng(BOOTSTRAP_SEED + 1)),
    }


def run_study(root: Path) -> dict:
    """Train every row on one 80/20 song split and score the holdout."""
    print(f"parsing scores under {root}")
    songs = iter_parsed_songs(root)
    if len(songs) < 40:
        raise RuntimeError(f"need at least 40 parsed songs, found {len(songs)} under {root}")
    rng = np.random.default_rng(SPLIT_SEED)
    hold_idx = holdout_indices(len(songs), rng, fraction=HOLDOUT_FRACTION)
    train_songs = [song for i, song in enumerate(songs) if i not in hold_idx]
    hold_songs = [song for i, song in enumerate(songs) if i in hold_idx]
    train_steps = [steps for song in train_songs if (steps := steps_for_song(song))]
    hold_steps = [steps for song in hold_songs if (steps := steps_for_song(song))]
    print(f"parsed {len(songs)} / {len(iter_mscx(root))} scores; train {len(train_steps)} hold {len(hold_steps)}")

    x, y = _stack(train_steps, ())
    tables = fit_interval_tables(train_steps)
    genre_std = float(np.std(x[:, GENRE])) if len(x) else 0.0
    arrangement_std = float(np.std(x[:, ARRANGEMENT])) if len(x) else 0.0

    models: list[tuple[str, tuple[int, ...], bool]] = [
        ("MLP, all features", (), False),
        ("MLP, lyric features removed", LYRIC_COLUMNS, False),
        ("MLP, phrase features removed", PHRASE_COLUMNS, False),
        ("MLP, previous degrees only", _drop_except(PITCH_COLUMNS), False),
        ("MLP + cadence teacher", (), True),
    ]
    fitted: list[tuple[str, dict[str, np.ndarray], tuple[int, ...], int]] = []
    for name, drop, mix_teacher in models:
        print(f"fitting {name}")
        xx, yy = _stack(train_steps, drop)
        if mix_teacher:
            from calliope.audio.vocal_melody_ml import _teacher_rows

            tx, ty = _teacher_rows(120, np.random.default_rng(SPLIT_SEED))
            extra_x = np.stack(tx).astype(np.float32)
            extra_y = np.stack(ty).astype(np.float32)
            xx = np.concatenate([xx, extra_x], axis=0)
            yy = np.concatenate([yy, extra_y], axis=0)
        weights, n_iter = fit_mlp_weights(xx, yy)
        fitted.append((name, weights, drop, n_iter))
        print(f"  n_iter={n_iter} rows={len(yy)}")

    rows: list[dict] = []

    def add_stat(name: str, predict) -> None:
        print(f"scoring {name}")
        scored = _eval_pair(hold_steps, predict)
        rows.append({"name": name, "kind": "baseline", **scored})

    add_stat(
        "Constant median",
        lambda steps, teacher: predict_constant(
            steps,
            int(tables["median_interval"]),
            lambda step: _beats_for(step, tables, positional=False),
            teacher=teacher,
        ),
    )
    add_stat(
        "Unigram mode",
        lambda steps, teacher: predict_constant(
            steps,
            int(tables["mode_interval"]),
            lambda step: _beats_for(step, tables, positional=False),
            teacher=teacher,
        ),
    )
    add_stat(
        "Markov interval",
        lambda steps, teacher: predict_markov(
            steps,
            tables,
            lambda step: _beats_for(step, tables, positional=False),
            teacher=teacher,
        ),
    )
    add_stat(
        "Markov interval + phrase duration",
        lambda steps, teacher: predict_markov(
            steps,
            tables,
            lambda step: _beats_for(step, tables, positional=True),
            teacher=teacher,
        ),
    )
    for name, weights, drop, n_iter in fitted:
        print(f"scoring {name}")
        scored = _eval_pair(
            hold_steps,
            lambda steps, teacher, weights=weights, drop=drop: predict_mlp(
                steps, weights, drop, teacher=teacher
            ),
        )
        rows.append({"name": name, "kind": "mlp", "n_iter": n_iter, "dropped": list(drop), **scored})

    full_weights = next(weights for name, weights, _drop, _n in fitted if name == "MLP, all features")
    full_row = next(row for row in rows if row["name"] == "MLP, all features")
    _check_against_evaluate_songs(hold_songs, full_weights, full_row)

    return {
        "corpus": "OpenScore Lieder",
        "corpus_version": "v3.0.0",
        "corpus_doi": "10.5281/zenodo.15450144",
        "license": "CC0",
        "n_mscx": len(iter_mscx(root)),
        "n_parsed_songs": len(songs),
        "n_train_songs": len(train_steps),
        "n_holdout_songs": len(hold_steps),
        "n_train_notes": int(sum(len(s) for s in train_steps)),
        "n_holdout_notes": int(sum(len(s) for s in hold_steps)),
        "split": {"seed": SPLIT_SEED, "fraction": HOLDOUT_FRACTION, "unit": "song"},
        "bootstrap": {"seed": BOOTSTRAP_SEED, "n": N_BOOTSTRAP, "unit": "song"},
        "feature_std": {"genre": genre_std, "arrangement": arrangement_std},
        "baselines": {
            "median_interval": int(tables["median_interval"]),
            "mode_interval": int(tables["mode_interval"]),
            "median_beats": float(tables["median_beats"]),
            "beats_by_bin": tables["beats_by_bin"],
        },
        "models": rows,
    }


def _drop_except(keep: tuple[int, ...]) -> tuple[int, ...]:
    keep_set = set(keep)
    return tuple(i for i in range(12) if i not in keep_set)


def _stack(songs: list[list[Step]], drop: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for steps in songs:
        x, y = matrix_from_steps(steps, drop)
        if len(x):
            xs.append(x)
            ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _check_against_evaluate_songs(
    hold_songs: list[ParsedSong],
    weights: dict[str, np.ndarray],
    full_row: dict,
) -> None:
    """Rollout interval MAE should match the trainer's own holdout metric."""
    from calliope.audio.vocal_melody_ml import evaluate_songs

    ref = evaluate_songs([song for song in hold_songs if steps_for_song(song)], weights)
    got = full_row["rollout"]
    for key, ref_key in (
        ("interval_mae", "interval_mae"),
        ("duration_mae", "duration_mae"),
        ("degree_acc", "scale_acc"),
    ):
        if abs(float(got[key]) - float(ref[ref_key])) > 1e-4:
            raise RuntimeError(
                f"rollout {key}={got[key]:.6f} disagrees with evaluate_songs "
                f"{ref_key}={ref[ref_key]:.6f}"
            )


def write_report_json(result: dict, path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def _fmt_table(result: dict, protocol: str) -> str:
    lines = [
        f"{'model':<40} {'interval':>10} {'95% CI':>18} {'duration':>10} {'contour':>10} {'degree':>10}",
    ]
    for row in result["models"]:
        m = row[protocol]
        lo, hi = m["interval_mae_ci95"]
        lines.append(
            f"{row['name']:<40} {m['interval_mae']:>10.3f} {f'[{lo:.3f}, {hi:.3f}]':>18} "
            f"{m['duration_mae']:>10.3f} {m['contour_acc']:>10.3f} {m['degree_acc']:>10.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[4] / "data" / "openscore-lieder"
    if not root.is_dir():
        raise SystemExit(f"missing corpus at {root}")
    result = run_study(root)
    dest = write_report_json(result)
    print(f"\nTeacher-forced (next note)\n{_fmt_table(result, 'teacher_forced')}")
    print(f"\nRollout (model history)\n{_fmt_table(result, 'rollout')}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
