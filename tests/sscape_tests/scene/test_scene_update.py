# SPDX-FileCopyrightText: (C) 2025 - 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json

from django.http import QueryDict
from django.test import TestCase
from rest_framework.exceptions import ValidationError

from manager.models import Scene
from manager.serializers import SceneSerializer


class SceneUpdateTestCase(TestCase):
  def setUp(self):
    self.scene = Scene.objects.create(name="test_scene")

  def test_create_scene_with_only_name(self):
    serializer = SceneSerializer(data=QueryDict("name=mapless_scene"))
    self.assertTrue(serializer.is_valid(), serializer.errors)

    scene = serializer.save()

    scene.refresh_from_db()
    self.assertEqual(scene.name, "mapless_scene")
    self.assertFalse(scene.map)
    self.assertFalse(serializer.data.get('map'))
    self.assertEqual(str(serializer.data['uid']), str(scene.pk))

  def test_mesh_transform_does_not_modify_request_data(self):
    data = {'mesh_translation': [1.0, 2.0, 3.0]}
    validated_data = {}

    SceneSerializer().handleMeshTransform(data, validated_data)

    self.assertEqual(data, {'mesh_translation': [1.0, 2.0, 3.0]})
    self.assertEqual(validated_data, {
      'translation_x': 1.0,
      'translation_y': 2.0,
      'translation_z': 3.0,
    })

  def test_invalid_mesh_transform_is_rejected(self):
    with self.assertRaises(ValidationError):
      SceneSerializer().handleMeshTransform({'mesh_translation': [1.0, 2.0]}, {})

  def test_map_corners_lla_json_string_is_parsed(self):
    corners = [
      [37.0, -122.0, 0],
      [37.1, -122.0, 0],
      [37.1, -121.9, 0],
      [37.0, -121.9, 0],
    ]
    serializer = SceneSerializer(
      instance=self.scene,
      data={
        'name': self.scene.name,
        'output_lla': True,
        'map_corners_lla': json.dumps(corners),
      },
      partial=True)
    self.assertTrue(serializer.is_valid(), serializer.errors)
    self.assertEqual(serializer.validated_data['map_corners_lla'], corners)

  def test_output_lla_accepts_lowercase_true_string(self):
    corners = [
      [37.0, -122.0, 0],
      [37.1, -122.0, 0],
      [37.1, -121.9, 0],
      [37.0, -121.9, 0],
    ]
    serializer = SceneSerializer(
      instance=self.scene,
      data={
        'name': self.scene.name,
        'output_lla': 'true',
        'map_corners_lla': json.dumps(corners),
      },
      partial=True)
    self.assertTrue(serializer.is_valid(), serializer.errors)
    self.assertIs(serializer.validated_data['output_lla'], True)

  def test_output_lla_rejects_unknown_choice(self):
    serializer = SceneSerializer(
      instance=self.scene,
      data={
        'name': self.scene.name,
        'output_lla': 'not-a-bool',
      },
      partial=True)
    self.assertFalse(serializer.is_valid())
    self.assertIn('output_lla', serializer.errors)

  def test_map_corners_lla_invalid_string_is_rejected(self):
    serializer = SceneSerializer(
      instance=self.scene,
      data={
        'name': self.scene.name,
        'map_corners_lla': 'not-json',
      },
      partial=True)
    self.assertFalse(serializer.is_valid())
    self.assertIn('map_corners_lla', serializer.errors)
