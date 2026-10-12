# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cam.save must not wipe transforms when scene_id is the same UUID as a str."""

from django.test import TestCase

from manager.models import Cam, Scene

TEST_NAME = "NEX-T28216"


class CamSavePreservesTransformsTestCase(TestCase):
  def test_str_scene_id_does_not_clear_transforms(self):
    scene = Scene.objects.create(name="cam_save_scene")
    cam = Cam.objects.create(
      sensor_id="save-cam",
      name="Save Cam",
      scene=scene,
      transform_type="3d-2d point correspondence",
      transforms=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )

    cam.scene_id = str(scene.pk)
    cam.transforms = [
      10.0,
      20.0,
      30.0,
      40.0,
      1.0,
      2.0,
      0.0,
      3.0,
      4.0,
      0.0,
    ]
    cam.save()
    cam.refresh_from_db()

    self.assertEqual(len(cam.transforms), 10)
    self.assertEqual(cam.transforms[0], 10.0)
    self.assertEqual(str(cam.scene_id), str(scene.pk))

  def test_real_scene_change_clears_unchanged_transforms(self):
    scene_a = Scene.objects.create(name="cam_save_a")
    scene_b = Scene.objects.create(name="cam_save_b")
    cam = Cam.objects.create(
      sensor_id="move-cam",
      name="Move Cam",
      scene=scene_a,
      transform_type="3d-2d point correspondence",
      transforms=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )

    cam.scene = scene_b
    cam.save()
    cam.refresh_from_db()

    self.assertEqual(cam.transforms, [])
    self.assertEqual(cam.scene_id, scene_b.pk)
