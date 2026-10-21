# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse
from rest_framework.authtoken.models import Token


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