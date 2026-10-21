# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from django.contrib.auth.models import User
from django.test import Client, SimpleTestCase, TestCase
from django.urls import reverse
from rest_framework.authtoken.models import Token
from rest_framework.test import APIRequestFactory, force_authenticate

from manager.api import GenerateSceneMesh, IsAdminOrReadOnly, ManageThing, SceneImportAPIView


class WorkspacePermissionTestCase(SimpleTestCase):
  def test_non_superuser_staff_cannot_generate_scene_mesh(self):
    request = APIRequestFactory().post('/api/v1/scene/test/generate-mesh/', {}, format='json')
    force_authenticate(request, user=User(username='viewer', is_staff=True, is_superuser=False))
    response = GenerateSceneMesh.as_view()(request, pk='00000000-0000-0000-0000-000000000001')
    self.assertEqual(response.status_code, 403)

  def test_read_only_users_and_staff_cannot_import(self):
    factory = APIRequestFactory()
    for staff in (False, True):
      with self.subTest(staff=staff):
        user = User(username='viewer', is_staff=staff, is_superuser=False)
        request = factory.post('/api/v1/import-scene/', {}, format='json')
        force_authenticate(request, user=user)
        self.assertEqual(SceneImportAPIView.as_view()(request).status_code, 403)

  def test_superuser_can_reach_import_validation(self):
    request = APIRequestFactory().post('/api/v1/import-scene/', {}, format='json')
    force_authenticate(request, user=User(username='editor', is_superuser=True))
    response = SceneImportAPIView.as_view()(request)
    self.assertEqual(response.status_code, 400)
    self.assertIn('zipFile', str(response.data))

  def test_anonymous_import_is_rejected(self):
    request = APIRequestFactory().post('/api/v1/import-scene/', {}, format='json')
    self.assertEqual(SceneImportAPIView.as_view()(request).status_code, 401)

  def test_read_only_users_cannot_write_scene_configuration(self):
    factory = APIRequestFactory()
    for method in ('POST', 'PUT', 'PATCH', 'DELETE'):
      with self.subTest(method=method):
        request = factory.generic(method, '/api/v1/scene', '{}', content_type='application/json')
        force_authenticate(request, user=User(username='viewer', is_superuser=False))
        self.assertEqual(ManageThing.as_view()(request, 'scene').status_code, 403)

  def test_authenticated_viewer_retains_safe_read_access(self):
    permission = IsAdminOrReadOnly()
    for method in ('GET', 'HEAD', 'OPTIONS'):
      with self.subTest(method=method):
        request = SimpleNamespace(method=method, user=User(username='viewer'))
        self.assertTrue(permission.has_permission(request, None))


class SessionExpiryTestCase(TestCase):
  def setUp(self):
    self.user = User.objects.create_user(username='session-user')
    self.url = reverse('session_status')

  def test_valid_session_returns_empty_noncacheable_success(self):
    self.client.force_login(self.user)
    response = self.client.get(self.url)
    self.assertEqual(response.status_code, 204)
    self.assertEqual(response.content, b'')
    self.assertEqual(response.headers['Cache-Control'], 'no-store')

  def test_second_browser_invalidates_first_without_revoking_second(self):
    first = Client()
    second = Client()
    first.force_login(self.user)
    self.assertEqual(first.get(self.url).status_code, 204)
    token, _ = Token.objects.get_or_create(user=self.user)
    second.force_login(self.user)
    self.assertEqual(first.get(self.url, HTTP_AUTHORIZATION=f'Token {token.key}').status_code, 401)
    first.get(reverse('sign_out'))
    self.assertEqual(second.get(self.url).status_code, 204)
    self.assertTrue(Token.objects.filter(pk=token.pk).exists())

  def test_anonymous_session_returns_401_instead_of_redirecting(self):
    response = self.client.get(self.url)
    self.assertEqual(response.status_code, 401)
    self.assertEqual(response.headers['Cache-Control'], 'no-store')

  def test_mutating_request_is_not_allowed(self):
    self.assertEqual(self.client.post(self.url).status_code, 405)