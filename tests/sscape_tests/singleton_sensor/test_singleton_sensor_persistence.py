# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json

from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TestCase, TransactionTestCase

from manager.models import Scene, SingletonSensor
from manager.serializers import SingletonSerializer


class SingletonSensorPersistenceTestCase(TestCase):
  def setUp(self):
    self.scene = Scene.objects.create(name="sensor_scene")

  def test_color_and_xyz_round_trip_with_independent_polygon(self):
    points = [[0, 0], [4, 0], [4, 3]]
    serializer = SingletonSerializer(data={
      'name': 'Infrared', 'sensor_id': 'infrared', 'scene': str(self.scene.pk),
      'area': 'poly', 'points': points, 'center': [15, 20],
      'translation': [15, 20, 3.5], 'color': '#12abEF',
    })
    self.assertTrue(serializer.is_valid(), serializer.errors)
    sensor = serializer.save()
    sensor.refresh_from_db()
    result = SingletonSerializer(sensor).data
    self.assertEqual(result['translation'], [15, 20, 3.5])
    self.assertEqual(result['center'], [15, 20])
    self.assertEqual(result['color'], '#12abEF')
    self.assertEqual([list(point) for point in result['points']], points)
    area = json.loads(sensor.areaJSON())
    self.assertEqual(area['z'], 3.5)
    self.assertEqual(area['color'], '#12abEF')

  def test_legacy_xy_update_preserves_height_and_color(self):
    sensor = SingletonSensor.objects.create(
      name='Legacy', sensor_id='legacy', scene=self.scene,
      map_x=1, map_y=2, map_z=3, color='#112233')
    serializer = SingletonSerializer(sensor, data={'center': [4, 5]}, partial=True)
    self.assertTrue(serializer.is_valid(), serializer.errors)
    serializer.save()
    sensor.refresh_from_db()
    self.assertEqual(SingletonSerializer(sensor).data['translation'], [4, 5, 3])
    self.assertEqual(sensor.color, '#112233')

  def test_existing_sensor_defaults_are_ground_level_and_purple(self):
    sensor = SingletonSensor.objects.create(name='Default', sensor_id='default', scene=self.scene)
    self.assertEqual(sensor.map_z, 0)
    self.assertEqual(sensor.color, '#bf5af2')

  def test_invalid_color_and_translation_are_rejected(self):
    for patch, field in [
      ({'color': 'red'}, 'color'),
      ({'translation': [1, 2]}, 'translation'),
      ({'translation': [1, 2, float('inf')]}, 'translation'),
      ({'translation': [1, 2, float('nan')]}, 'translation'),
    ]:
      with self.subTest(patch=patch):
        serializer = SingletonSerializer(data={'name': 'Invalid', **patch})
        self.assertFalse(serializer.is_valid())
        self.assertIn(field, serializer.errors)


class SingletonSensorMigrationTestCase(TransactionTestCase):
  def test_existing_sensor_coordinates_survive_schema_upgrade(self):
    previous = [('manager', '0010_boundingbox_color_alter_pubsubacl_topic')]
    current = [('manager', '0011_sensor_color_and_height')]
    executor = MigrationExecutor(connection)
    try:
      executor.migrate(previous)
      old_apps = executor.loader.project_state(previous).apps
      old_scene = old_apps.get_model('manager', 'Scene').objects.create(name='Existing scene')
      old_sensor = old_apps.get_model('manager', 'SingletonSensor').objects.create(
        name='Existing sensor', sensor_id='existing', scene_id=old_scene.pk,
        map_x=12.5, map_y=-3.25)
      executor = MigrationExecutor(connection)
      executor.migrate(current)
      new_apps = executor.loader.project_state(current).apps
      sensor = new_apps.get_model('manager', 'SingletonSensor').objects.get(pk=old_sensor.pk)
      self.assertEqual((sensor.map_x, sensor.map_y), (12.5, -3.25))
      self.assertEqual(sensor.map_z, 0)
      self.assertEqual(sensor.color, '#bf5af2')
    finally:
      MigrationExecutor(connection).migrate(current)