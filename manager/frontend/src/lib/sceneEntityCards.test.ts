// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { cameraSceneAfterSave, childCardFromRest } from "./sceneEntityCards";

const scenes = [
  { id: "17efd356-8858-4bf7-80fa-5bc58b2175f8", name: "Parking Lot" },
];

describe("camera save destination", () => {
  it("opens the scene for a newly created camera", () => {
    expect(cameraSceneAfterSave("destination", null, true)).toBe("/destination/");
    expect(cameraSceneAfterSave("current", "current", true)).toBe("/current/");
  });
  it("opens a scene when assigning an orphan or moving an existing camera", () => {
    expect(cameraSceneAfterSave("destination", null, false)).toBe("/destination/");
    expect(cameraSceneAfterSave("destination", "source", false)).toBe("/destination/");
  });
  it("does not navigate for ordinary same-scene edits or no destination", () => {
    expect(cameraSceneAfterSave("current", "current", false)).toBeNull();
    expect(cameraSceneAfterSave("", null, true)).toBeNull();
  });
});

describe("childCardFromRest", () => {
  it("keys a local child by its scene uuid, not the link pk", () => {
    const card = childCardFromRest(
      {
        uid: 1,
        child_type: "local",
        child: "17efd356-8858-4bf7-80fa-5bc58b2175f8",
      },
      scenes,
    );
    expect(card).not.toBeNull();
    expect(card?.id).toBe("17efd356-8858-4bf7-80fa-5bc58b2175f8");
    expect(card?.id).not.toBe("1");
    expect(card?.restUid).toBe(card?.id);
    expect(card?.deleteUrl).toBe("/child/delete/1/");
  });

  it("keys a remote child by remote_child_id", () => {
    const remote = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
    const card = childCardFromRest(
      {
        uid: "2",
        child_type: "remote",
        remote_child_id: remote,
        child_name: "Remote annex",
      },
      scenes,
    );
    expect(card?.id).toBe(remote);
    expect(card?.deleteUrl).toBe("/child/delete/2/");
  });

  it("returns null when the payload has no uid", () => {
    expect(childCardFromRest({ child_type: "local" }, scenes)).toBeNull();
  });
});
