// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react";
import { upsertRoiMeta, upsertTripMeta } from "./map/geometryModel";
import {
  roiFromLoad,
  type RoiLoadJson,
  type TripwireLoadJson,
} from "./roiTypes";

type Props = {
  sceneId: string;
  initialRegions: RoiLoadJson[];
  initialTripwires: TripwireLoadJson[];
};

/**
 * Seeds the typed geometry model from the Django bootstrap payload.
 * Replaces the initialization half of the legacy RoiTripwireEditors —
 * the 3D viewport reads regions/tripwires from this model.
 * Renders nothing.
 */
export function GeometryBootstrap({
  sceneId,
  initialRegions,
  initialTripwires,
}: Props) {
  useEffect(() => {
    initialRegions.forEach((raw) => {
      const uuid = String(raw.uuid || "").trim();
      if (!uuid) {
        return;
      }
      const roi = roiFromLoad(raw, sceneId);
      if (!roi) {
        return;
      }
      const sectors = [
        { color: "green", color_min: roi.greenMin },
        { color: "yellow", color_min: roi.yellowMin },
        { color: "red", color_min: roi.redMin },
      ];
      upsertRoiMeta(roi.uuid, {
        title: roi.title,
        volumetric: roi.volumetric,
        height: roi.height,
        buffer_size: roi.buffer_size,
        range_max: roi.rangeMax,
        sectors,
        points: (raw.points || []).map(
          (p) => [Number(p[0]), Number(p[1])] as [number, number],
        ),
      });
    });
    initialTripwires.forEach((raw) => {
      const uuid = String(raw.uuid || "").trim();
      if (!uuid) {
        return;
      }
      upsertTripMeta(uuid, {
        title: (raw.title || "").trim(),
        points: (raw.points || []).map(
          (p) => [Number(p[0]), Number(p[1])] as [number, number],
        ),
      });
    });
  }, [initialRegions, initialTripwires, sceneId]);

  return null;
}
