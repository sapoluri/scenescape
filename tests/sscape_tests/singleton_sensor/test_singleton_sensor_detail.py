# SPDX-FileCopyrightText: (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from django.test import TestCase
from django.urls import reverse
from manager.models import SingletonSensor, Scene
from django.contrib.auth.models import User
from django.test.client import RequestFactory

class SingletonSensorDetailTestCase(TestCase):
  def setUp(self):
    self.factory = RequestFactory()
    request = self.factory.get('/')
    self.user = User.objects.create_superuser('test_user', 'test_user@intel.com', 'testpassword')
    self.client.post(reverse('sign_in'), data = {'username': 'test_user', 'password': 'testpassword', 'request': request})
    testScene = Scene.objects.create(name = "test_scene")
    SingletonSensor.objects.create(sensor_id="100", name="test_camera", scene = testScene)

  def test_singleton_sensor_detail_page(self):
    response = self.client.get('/singleton_sensor/calibrate/1')
    self.assertEqual(response.status_code, 200)
