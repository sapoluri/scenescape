# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Manager ui-bootstrap payloads (chrome off-k8s regression)."""

import uuid

from django.contrib.auth.models import AnonymousUser, User
from django.test import RequestFactory, TestCase, override_settings
from django.urls import reverse
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient

from manager.models import Cam, ChildScene, Scene, SingletonSensor
from manager.services.ui_bootstrap import (
  advertise_broker_host,
  build_chrome_bootstrap,
  build_models_directory_bootstrap,
  build_scene_detail_bootstrap,
  build_scenes_home_bootstrap,
  user_auth_token,
)

TEST_NAME = "NEX-T18750"


@override_settings(KUBERNETES_SERVICE_HOST=False)
class ChromeBootstrapOffKubernetesTests(TestCase):
  """Compose demos are not Kubernetes; chrome must not reverse model_list."""

  def setUp(self):
    self.factory = RequestFactory()
    self.user = User.objects.create_superuser(
      "chrome_user", "chrome@example.com", "testpassword",
    )

  def test_build_chrome_bootstrap_omits_models_url(self):
    request = self.factory.get("/")
    request.user = self.user
    payload = build_chrome_bootstrap(request)
    self.assertFalse(payload["isKubernetes"])
    self.assertNotIn("models", payload["urls"])
    self.assertEqual(payload["urls"]["cameras"], reverse("cam_list"))

  def test_ui_bootstrap_chrome_api_returns_200(self):
    client = APIClient()
    client.force_authenticate(user=self.user)
    response = client.get("/api/v1/ui-bootstrap/", {"page": "chrome"})
    self.assertEqual(response.status_code, 200)
    self.assertNotIn("models", response.data["urls"])
    self.assertTrue(response.data["authenticated"])

  def test_ui_bootstrap_chrome_anonymous_ok(self):
    client = APIClient()
    response = client.get("/api/v1/ui-bootstrap/", {"page": "chrome"})
    self.assertEqual(response.status_code, 200)
    self.assertFalse(response.data["authenticated"])
    self.assertNotIn("models", response.data["urls"])


@override_settings(KUBERNETES_SERVICE_HOST=False)
class AuthTokenProvisionTests(TestCase):
  def test_user_auth_token_creates_missing_token(self):
    user = User.objects.create_user("tok_user", "tok@example.com", "pw")
    self.assertFalse(Token.objects.filter(user=user).exists())
    key = user_auth_token(user)
    self.assertTrue(key)
    self.assertTrue(Token.objects.filter(user=user, key=key).exists())

  def test_user_auth_token_anonymous_empty(self):
    self.assertEqual(user_auth_token(AnonymousUser()), "")
    self.assertEqual(user_auth_token(None), "")


@override_settings(KUBERNETES_SERVICE_HOST=False)
class ModelsDirectoryBootstrapTests(TestCase):
  def setUp(self):
    self.factory = RequestFactory()
    self.user = User.objects.create_superuser(
      "models_user", "models@example.com", "testpassword",
    )

  def test_models_bootstrap_includes_auth_token(self):
    request = self.factory.get("/")
    request.user = self.user
    payload = build_models_directory_bootstrap(request)
    self.assertTrue(payload["isSuperuser"])
    self.assertTrue(payload["authToken"])
    self.assertEqual(payload["authToken"], user_auth_token(self.user))


@override_settings(KUBERNETES_SERVICE_HOST=False)
class SceneBootstrapIdentityTests(TestCase):
  """Child viewport ids are scene UUIDs, and gallery counts split cameras."""

  def setUp(self):
    self.factory = RequestFactory()
    self.user = User.objects.create_superuser(
      "scene_boot", "scene_boot@example.com", "testpassword",
    )
    self.parent = Scene.objects.create(name="Retail Floor")
    self.local_child = Scene.objects.create(name="Parking Lot")
    self.link = ChildScene.objects.create(
      parent=self.parent, child=self.local_child, child_type="local",
    )
    self.remote_id = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    self.remote = ChildScene.objects.create(
      parent=self.parent,
      child=None,
      child_name="Remote annex",
      child_type="remote",
      remote_child_id=self.remote_id,
    )
    self.camera = Cam.objects.create(
      sensor_id="cam-1",
      name="Entrance Cam",
      scene=self.parent,
      type="camera",
    )
    SingletonSensor.objects.create(
      sensor_id="sen-1",
      name="Lobby temperature",
      scene=self.parent,
      type="generic",
    )

  @override_settings(ALLOWED_HOSTS=["10.0.0.5", "testserver", "localhost"])
  def test_broker_host_prefers_request_ip(self):
    request = self.factory.get("/", HTTP_HOST="10.0.0.5")
    request.user = self.user
    self.assertEqual(advertise_broker_host(request), "10.0.0.5")
    payload = build_scene_detail_bootstrap(request, self.parent.id)
    self.assertEqual(payload["brokerHost"], "10.0.0.5")

  @override_settings(ALLOWED_HOSTS=["localhost", "192.168.1.50", "testserver"])
  def test_broker_host_uses_allowed_hosts_ip_when_loopback(self):
    request = self.factory.get("/", HTTP_HOST="localhost")
    request.user = self.user
    self.assertEqual(advertise_broker_host(request), "192.168.1.50")

  def test_child_id_is_scene_uuid_not_link_pk(self):
    request = self.factory.get("/")
    request.user = self.user
    payload = build_scene_detail_bootstrap(request, self.parent.id)
    by_name = {row["name"]: row for row in payload["children"]}
    local = by_name["Parking Lot"]
    remote = by_name["Remote annex"]
    self.assertEqual(local["id"], str(self.local_child.id))
    self.assertEqual(local["restUid"], str(self.local_child.id))
    self.assertNotEqual(local["id"], str(self.link.id))
    self.assertNotEqual(local["id"], str(self.camera.id))
    self.assertIn(f"/child/delete/{self.link.id}/", local["deleteUrl"])
    self.assertEqual(remote["id"], str(self.remote_id))
    self.assertNotEqual(remote["id"], str(self.remote.id))

  def test_home_counts_do_not_treat_cameras_as_sensors(self):
    request = self.factory.get("/")
    request.user = self.user
    payload = build_scenes_home_bootstrap(request)
    card = next(s for s in payload["scenes"] if s["name"] == "Retail Floor")
    self.assertEqual(card["counts"]["cameras"], 1)
    self.assertEqual(card["counts"]["sensors"], 1)

  def test_child_without_uuid_falls_back_to_link_pk(self):
    bare = ChildScene.objects.create(
      parent=self.parent,
      child=None,
      child_name="Unassigned",
      child_type="remote",
      remote_child_id=None,
    )
    request = self.factory.get("/")
    request.user = self.user
    payload = build_scene_detail_bootstrap(request, self.parent.id)
    row = next(c for c in payload["children"] if c["name"] == "Unassigned")
    self.assertEqual(row["id"], str(bare.id))
    self.assertEqual(row["restUid"], str(bare.id))
