// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import {
  applyChildPoseToObject,
  parseChildPose,
} from "./childPose";

describe("parseChildPose", () => {
  it("reads nested transform translation/rotation/scale", () => {
    const pose = parseChildPose({
      transform: {
        translation: [1, 2, 3],
        rotation: [10, 20, 30],
        scale: [2, 2, 2],
      },
    });
    expect(pose.position).toEqual([1, 2, 3]);
    expect(pose.rotation).toEqual([10, 20, 30]);
    expect(pose.scale).toEqual([2, 2, 2]);
  });

  it("reads flat transform1..9 euler layout", () => {
    const pose = parseChildPose({
      transform1: 4,
      transform2: 5,
      transform3: 6,
      transform4: 7,
      transform5: 8,
      transform6: 9,
      transform7: 1.5,
      transform8: 1.5,
      transform9: 1.5,
    });
    expect(pose.position).toEqual([4, 5, 6]);
    expect(pose.rotation).toEqual([7, 8, 9]);
    expect(pose.scale).toEqual([1.5, 1.5, 1.5]);
  });

  it("defaults to identity when fields are missing", () => {
    expect(parseChildPose({})).toEqual({
      position: [0, 0, 0],
      rotation: [0, 0, 0],
      scale: [1, 1, 1],
    });
  });
});

describe("applyChildPoseToObject", () => {
  it("writes meters and degrees→radians onto a triple target", () => {
    const object = {
      position: { x: 0, y: 0, z: 0, set(x: number, y: number, z: number) {
        this.x = x;
        this.y = y;
        this.z = z;
      } },
      rotation: { x: 0, y: 0, z: 0, set(x: number, y: number, z: number) {
        this.x = x;
        this.y = y;
        this.z = z;
      } },
      scale: { x: 1, y: 1, z: 1, set(x: number, y: number, z: number) {
        this.x = x;
        this.y = y;
        this.z = z;
      } },
    };
    applyChildPoseToObject(object, {
      position: [3, 4, 5],
      rotation: [90, 0, 0],
      scale: [2, 3, 4],
    });
    expect(object.position.x).toBe(3);
    expect(object.position.y).toBe(4);
    expect(object.position.z).toBe(5);
    expect(object.rotation.x).toBeCloseTo(Math.PI / 2, 9);
    expect(object.scale.x).toBe(2);
    expect(object.scale.y).toBe(3);
    expect(object.scale.z).toBe(4);
  });
});
