# SPDX-FileCopyrightText: (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from django.test import TestCase
from django.urls import reverse
from django.contrib.auth.models import User
from django.test.client import RequestFactory
from django.test.client import RequestFactory

class SceneCreateTestCase(TestCase):
  def setUp(self):
    self.factory = RequestFactory()
    request = self.factory.get('/')
    self.user = User.objects.create_superuser('test_user', 'test_user@intel.com', 'testpassword')
    self.client.post(reverse('sign_in'), data = {'username': 'test_user', 'password': 'testpassword', 'request': request})

  def test_scene_create_page(self):
    response = self.client.get(reverse('scene_create'), data = {'name': 'test_scene'})
    self.assertEqual(response.status_code, 200)
    self.assertTemplateUsed(response, 'scene/scene_create.html')
