# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from django.test import TestCase
from unittest.mock import patch

from manager.models import Cam, Scene
from manager.serializers import CamSerializer
from scene_common.options import QUATERNION
from scene_common.scenescape import SceneLoader

# [tx, ty, tz, qx, qy, qz, qw, sx, sy, sz]
INITIAL_TRANSFORMS = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
TRANSLATION_DELTA = 5.0


class SceneCameraCacheTestCase(TestCase):
  """Scene cache must reflect camera pose changes from the database."""

  def setUp(self):
    SceneLoader.scenes.clear()
    self.scene = Scene.objects.create(name="camera_cache_scene", scale=100.0)
    self.cam = Cam.objects.create(
      sensor_id="camera_cache_cam",
      name="Camera Cache Cam",
      type="camera",
      scene=self.scene,
      transforms=list(INITIAL_TRANSFORMS),
      transform_type=QUATERNION,
    )
    return

  def tearDown(self):
    SceneLoader.scenes.clear()
    return

  def cachedTranslation(self):
    camera = self.scene.scenescapeScene.cameraWithID(self.cam.sensor_id)
    return camera.pose.translation.asNumpyCartesian.tolist()

  def test_cached_pose_reflects_saved_translation(self):
    """Positive: a saved pose change is visible through the cached scene."""
    self.cachedTranslation()

    transforms = list(self.cam.transforms)
    transforms[0] += TRANSLATION_DELTA
    transforms[1] += TRANSLATION_DELTA
    self.cam.transforms = transforms
    self.cam.save()

    persisted = Cam.objects.get(pk=self.cam.pk).transforms[:3]
    self.assertEqual(persisted, transforms[:3],
                     "precondition failed: the pose was not persisted")

    self.assertEqual(self.cachedTranslation(), persisted,
                     "cached scene returned a pose that no longer matches the database")
    return

  def test_cached_pose_stable_without_save(self):
    """Negative: with no intervening save, repeated reads must not change."""
    first = self.cachedTranslation()
    self.assertEqual(self.cachedTranslation(), first)
    self.assertEqual(first, INITIAL_TRANSFORMS[:3])
    return

  def test_serialized_pose_survives_a_concurrent_cache_eviction(self):
    cached_scene = self.scene.scenescapeScene
    with patch.object(cached_scene, 'cameraWithID', return_value=None):
      data = CamSerializer(self.cam).data
    self.assertEqual(data.get('translation'), INITIAL_TRANSFORMS[:3])
    self.assertEqual(list(data['rotation']), [0.0, 0.0, 0.0])
    self.assertEqual(list(data['scale']), [1.0, 1.0, 1.0])

  def test_serialization_leaves_shared_camera_cache_untouched(self):
    cached_scene = self.scene.scenescapeScene
    cached_camera = cached_scene.cameras[self.cam.sensor_id]
    data = CamSerializer(self.cam).data
    self.assertEqual(data['translation'], INITIAL_TRANSFORMS[:3])
    self.assertIs(cached_scene.cameras[self.cam.sensor_id], cached_camera)

  def test_pose_is_computed_once_for_all_serialized_fields(self):
    from scene_common.camera import Camera
    with patch('manager.serializers.ScenescapeCamera', wraps=Camera) as camera:
      serializer = CamSerializer(self.cam)
      serializer.data
    camera.assert_called_once()

  def test_point_correspondence_pose_matches_existing_scene_conversion(self):
    self.cam.transform_type = '3d-2d point correspondence'
    self.cam.transforms = [
      640, 360, 992, 360, 992, 712, 640, 712,
      0, 0, 0, 4, 0, 0, 4, 4, 0, 0, 4, 0,
    ]
    self.cam.intrinsics_fx = self.cam.intrinsics_fy = 880
    self.cam.intrinsics_cx = 640
    self.cam.intrinsics_cy = 360
    self.cam.width, self.cam.height = 1280, 720
    self.cam.save()
    expected = self.scene.scenescapeScene.cameraWithID(self.cam.sensor_id).pose
    actual = CamSerializer(self.cam).data
    for observed, saved in zip(actual['translation'], expected.translation.asNumpyCartesian):
      self.assertAlmostEqual(observed, saved, places=5)
    for observed, saved in zip(actual['rotation'], expected.euler_rotation):
      self.assertAlmostEqual(observed, saved, places=5)

  def test_camera_pose_reset_on_scene_reassignment(self):
    """Positive: camera pose must be cleared when reassigned to a different scene."""
    self.assertEqual(self.cam.transforms, INITIAL_TRANSFORMS)
    self.assertIsNone(self.cam.scene_x)
    self.assertIsNone(self.cam.scene_y)
    self.assertIsNone(self.cam.scene_z)

    new_scene = Scene.objects.create(name="new_scene", scale=100.0)

    self.cam.scene = new_scene
    self.cam.save()

    persisted = Cam.objects.get(pk=self.cam.pk)
    self.assertEqual(persisted.transforms, [],
                     "transforms should be cleared when scene is reassigned")
    self.assertIsNone(persisted.scene_x,
                      "scene_x should be None after reassignment")
    self.assertIsNone(persisted.scene_y,
                      "scene_y should be None after reassignment")
    self.assertIsNone(persisted.scene_z,
                      "scene_z should be None after reassignment")
    return
