// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

type Position = readonly [number, number, number];
export const TRACK_SMOOTHING_ALPHA = 2 / (7 + 1);
const PREFERENCE_KEY = "ss-viewer-smooth-tracks";

export function readSmoothTracksPreference(): boolean {
  try {
    return localStorage.getItem(PREFERENCE_KEY) === "1";
  } catch {
    return false;
  }
}

export function writeSmoothTracksPreference(enabled: boolean): void {
  try {
    localStorage.setItem(PREFERENCE_KEY, enabled ? "1" : "0");
  } catch {
    return;
  }
}

export function createTrackSmoother() {
  const tracks = new Map<string, { sample: Position; position: Position }>();
  return {
    update(id: string, sample: Position): Position {
      const previous = tracks.get(id);
      if (previous?.sample === sample) return previous.position;
      const position: Position = previous
        ? [
            previous.position[0] + TRACK_SMOOTHING_ALPHA * (sample[0] - previous.position[0]),
            previous.position[1] + TRACK_SMOOTHING_ALPHA * (sample[1] - previous.position[1]),
            previous.position[2] + TRACK_SMOOTHING_ALPHA * (sample[2] - previous.position[2]),
          ]
        : [...sample];
      tracks.set(id, { sample, position });
      return position;
    },
    retain(ids: ReadonlySet<string>): void {
      for (const id of tracks.keys()) {
        if (!ids.has(id)) tracks.delete(id);
      }
    },
    clear(): void {
      tracks.clear();
    },
  };
}