export const GESTURES_STUDIO_IMPORT_KEY = "calliope.gesturesStudioImport";

export type GesturesStudioImport = {
  sessionId: string;
  recordingId: string;
  name: string;
  durationSec: number;
  scoreLabel?: string;
};

export function stashGesturesStudioImport(payload: GesturesStudioImport): void {
  sessionStorage.setItem(GESTURES_STUDIO_IMPORT_KEY, JSON.stringify(payload));
}

export function peekGesturesStudioImport(): GesturesStudioImport | null {
  const raw = sessionStorage.getItem(GESTURES_STUDIO_IMPORT_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as GesturesStudioImport;
  } catch {
    sessionStorage.removeItem(GESTURES_STUDIO_IMPORT_KEY);
    return null;
  }
}

export function clearGesturesStudioImport(): void {
  sessionStorage.removeItem(GESTURES_STUDIO_IMPORT_KEY);
}

/** @deprecated Use peek + delayed clear so React Strict Mode remounts still see the payload. */
export function takeGesturesStudioImport(): GesturesStudioImport | null {
  const parsed = peekGesturesStudioImport();
  if (parsed) clearGesturesStudioImport();
  return parsed;
}
