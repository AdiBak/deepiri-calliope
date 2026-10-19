from pathlib import Path

import numpy as np

from calliope.audio.lieder_melody import lieder_feature_rows, parse_mscx_song
from calliope.audio.melody_eval import (
    LYRIC_COLUMNS,
    PHRASE_COLUMNS,
    Step,
    apply_drop,
    duration_bin,
    fit_interval_tables,
    matrix_from_steps,
    predict_constant,
    score_song,
    steps_for_song,
)
from calliope.audio.vocal_melody_ml import holdout_indices, syllable_features

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lieder_snippet.mscx"


def _line(deltas: list[int], beats: float = 0.5) -> list[Step]:
    pool = 24
    idx = pool // 2
    steps: list[Step] = []
    n = len(deltas)
    for i, delta in enumerate(deltas):
        idx = idx + delta
        steps.append(
            Step(
                tok="la",
                i=i,
                n=n,
                phrase_idx=0,
                n_phrases=1,
                phrase_end=i == n - 1,
                phrase_start=i == 0,
                true_idx=idx,
                beats=beats,
                phrase_pos=0.0 if n == 1 else i / (n - 1),
                pool_len=pool,
                lo=48,
                hi=84,
            )
        )
    return steps


def test_holdout_indices_match_the_trainer_split():
    rng = np.random.default_rng(7)
    expected = {int(i) for i in rng.permutation(100)[:5]}
    assert holdout_indices(100, np.random.default_rng(7)) == expected
    assert holdout_indices(30, np.random.default_rng(7)) == set()


def test_report_split_holds_out_a_fifth_of_songs():
    rng = np.random.default_rng(7)
    n_hold = int(round(0.2 * 100))
    expected = {int(i) for i in rng.permutation(100)[:n_hold]}
    assert holdout_indices(100, np.random.default_rng(7), fraction=0.2) == expected


def test_steps_match_training_rows():
    song = parse_mscx_song(_FIXTURE)
    assert song is not None
    steps = steps_for_song(song)
    got_x, got_y = matrix_from_steps(steps)
    fx, fy = lieder_feature_rows(song.notes, root=song.root, scale=song.scale)
    assert np.allclose(got_x, np.stack(fx))
    assert np.allclose(got_y, np.stack(fy))


def test_feature_drop_zeros_only_those_columns():
    song = parse_mscx_song(_FIXTURE)
    assert song is not None
    steps = steps_for_song(song)
    full, _y = matrix_from_steps(steps)
    dropped, _y = matrix_from_steps(steps, LYRIC_COLUMNS + PHRASE_COLUMNS)
    assert np.all(dropped[:, list(LYRIC_COLUMNS + PHRASE_COLUMNS)] == 0.0)
    kept = [i for i in range(12) if i not in LYRIC_COLUMNS + PHRASE_COLUMNS]
    assert np.allclose(dropped[:, kept], full[:, kept])


def test_vowel_column_is_the_documented_index():
    feat = syllable_features(
        tok="ii",
        i=0,
        n=4,
        phrase_idx=0,
        n_phrases=1,
        phrase_end=False,
        phrase_start=True,
        genre_preset="acoustic",
        arrangement_style="through-composed",
        prev_idx=0,
        prev2_idx=0,
        pool_len=8,
        lo=60,
        hi=72,
    )
    masked = apply_drop(feat, LYRIC_COLUMNS)
    assert feat[4] == 0.4  # vowel "i"
    assert masked[4] == 0.0
    assert masked[5] == 0.0
    assert masked[3] == feat[3]


def test_constant_step_has_zero_interval_error_when_the_line_matches():
    steps = _line([1, 1, 1, 1, 1, 1])
    idxs, beats = predict_constant(steps, 1, lambda _s: 0.5, teacher=True)
    scored = score_song(steps, idxs, beats, teacher=True)
    assert scored.interval_mae == 0.0
    assert scored.contour_hits == scored.n
    assert scored.degree_hits == scored.n


def test_duration_bin_prefers_the_phrase_ending():
    step = _line([0, 0, 1])[-1]
    assert step.phrase_end
    assert duration_bin(step) == "end"


def test_interval_tables_recover_a_repeated_step():
    tables = fit_interval_tables([_line([1, 1, 1, 1, 1, 1])])
    assert tables["median_interval"] == 1
    assert tables["mode_interval"] == 1
    assert tables["markov"][1] == 1
