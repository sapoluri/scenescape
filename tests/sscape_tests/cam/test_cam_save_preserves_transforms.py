# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cam.save must not wipe transforms when scene_id is the same UUID as a str."""

from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework.test import APIRequestFactory, force_authenticate

from manager.api import ManageThing
from manager.models import Cam, Scene
from manager.serializers import CamSerializer

TEST_NAME = "NEX-T28216"


class CamSavePreservesTransformsTestCase(TestCase):
  def test_create_api_returns_created_with_resolution_and_supported_distortion(self):
    scene = Scene.objects.create(name="api_camera_scene")
    request = APIRequestFactory().post('/api/v1/camera', {
      'name': 'API Camera', 'sensor_id': 'api_camera', 'scene': str(scene.pk),
      'transform_type': 'euler', 'translation': [1, 2, 3],
      'rotation': [30, 0, 45], 'scale': [1, 1, 1],
      'intrinsics': {'fx': 880, 'fy': 880, 'cx': 640, 'cy': 360},
      'distortion': {'k1': 0.1, 'k2': 0, 'p1': 0, 'p2': 0, 'k3': 0},
      'resolution': {'width': 1280, 'height': 720},
    }, format='json')
    force_authenticate(request, user=User(username='test-admin', is_superuser=True))
    response = ManageThing.as_view()(request, thing_type='camera')
    self.assertEqual(response.status_code, 201, response.data)
    self.assertEqual(response.data['resolution'], [1280, 720])
    self.assertEqual(response.data['translation'], [1, 2, 3])
    self.assertAlmostEqual(Cam.objects.get(sensor_id='api_camera').distortion_k1, 0.1)

  def test_camera_scene_reassignment_preserves_identity(self):
    source = Scene.objects.create(name='camera_move_source')
    destination = Scene.objects.create(name='camera_move_destination')
    camera = Cam.objects.create(name='Moved Camera', sensor_id='moved_camera', scene=source)
    original_pk = camera.pk
    serializer = CamSerializer(camera, data={'scene': str(destination.pk)}, partial=True)
    self.assertTrue(serializer.is_valid(), serializer.errors)
    serializer.save()
    camera.refresh_from_db()
    self.assertEqual(camera.pk, original_pk)
    self.assertEqual(camera.scene_id, destination.pk)
    self.assertFalse(Cam.objects.filter(scene=source, sensor_id='moved_camera').exists())

  def test_invalid_resolution_is_rejected_instead_of_raising_500(self):
    for resolution in ({'width': 0, 'height': 720}, {'width': 1280}, [1280]):
      with self.subTest(resolution=resolution):
        serializer = CamSerializer(data={'name': 'Invalid', 'resolution': resolution})
        self.assertFalse(serializer.is_valid())
        self.assertIn('resolution', serializer.errors)

  def test_legacy_resolution_list_is_supported(self):
    serializer = CamSerializer(data={
      'name': 'Legacy Resolution', 'sensor_id': 'legacy_resolution', 'resolution': [640, 480],
    })
    self.assertTrue(serializer.is_valid(), serializer.errors)
    camera = serializer.save()
    self.assertEqual((camera.width, camera.height), (640, 480))

  def test_create_placed_camera_with_pose_and_optics(self):
    scene = Scene.objects.create(name="placed_camera_scene")
    serializer = CamSerializer(data={
      'name': 'Placed Camera', 'sensor_id': 'placed_camera', 'scene': str(scene.pk),
      'transform_type': 'euler', 'translation': [1, 2, 2.5],
      'rotation': [30, 0, 45], 'scale': [1, 1, 1],
      'intrinsics': {'fx': 880, 'fy': 880, 'cx': 640, 'cy': 360},
      'distortion': {'k1': 0, 'k2': 0, 'p1': 0, 'p2': 0, 'k3': 0},
      'resolution': {'width': 1280, 'height': 720},
    })
    self.assertTrue(serializer.is_valid(), serializer.errors)
    camera = serializer.save()
    camera.refresh_from_db()
    self.assertEqual(str(camera.scene_id), str(scene.pk))
    self.assertEqual(serializer.data['name'], 'Placed Camera')
    self.assertEqual(serializer.data['translation'], [1, 2, 2.5])

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
