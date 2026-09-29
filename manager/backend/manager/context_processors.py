# SPDX-FileCopyrightText: (C) 2021 - 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.urls import reverse


def _auth_token_for_user(user):
  if user is None or not getattr(user, "is_authenticated", False):
    return ""
  try:
    token = user.auth_token
    return str(token) if token else ""
  except ObjectDoesNotExist:
    return ""


def _chrome_active_nav(request):
  match = getattr(request, "resolver_match", None)
  name = getattr(match, "url_name", None) if match else None
  if name in ("index", "sceneDetail"):
    return "scenes"
  if name in ("cam_list", "cam_create", "cam_update", "cam_calibrate", "cam_delete"):
    return "cameras"
  if name in (
      "singleton_sensor_list",
      "singleton_sensor_create",
      "singleton_sensor_update",
      "singleton_sensor_delete",
  ):
    return "sensors"
  if name == "model_list":
    return "models"
  if name in ("asset_list", "asset_create", "asset_update", "asset_delete"):
    return "assets"
  return None


def chrome_bootstrap(request):
  """JSON payload for the React chrome island (`ss-chrome-bootstrap`)."""
  user = getattr(request, "user", None)
  authenticated = bool(user and getattr(user, "is_authenticated", False))
  docs_version = settings.DOCS_VERSION
  return {
    "authenticated": authenticated,
    "username": user.username if authenticated else "",
    "isStaff": bool(authenticated and getattr(user, "is_staff", False)),
    "isKubernetes": bool(settings.KUBERNETES_SERVICE_HOST),
    "appName": settings.APP_PROPER_NAME,
    "appVersion": settings.APP_VERSION_NUMBER,
    "appGitCommit": settings.APP_GIT_COMMIT,
    "docsVersion": docs_version,
    "urls": {
      "home": "/",
      "scenes": "/",
      "cameras": reverse("cam_list"),
      "sensors": reverse("singleton_sensor_list"),
      "models": reverse("model_list"),
      "assets": reverse("asset_list"),
      "admin": "/admin",
      "signOut": "/sign_out",
      "docs": (
        f"https://docs.openedgeplatform.intel.com/{docs_version}"
        "/scenescape/index.html"
      ),
      "support": "https://github.com/open-edge-platform/scenescape/issues",
      "intel": "https://www.intel.com/",
      "intelLogo": f"{settings.STATIC_URL}images/intel-logo.svg",
    },
    "activeNav": _chrome_active_nav(request),
  }


def selected_settings(request):
  return {
    'APP_VERSION_NUMBER': settings.APP_VERSION_NUMBER,
    'APP_GIT_COMMIT': settings.APP_GIT_COMMIT,
    'DOCS_VERSION': settings.DOCS_VERSION,
    'APP_PROPER_NAME': settings.APP_PROPER_NAME,
    'APP_BASE_NAME': settings.APP_BASE_NAME,
    'KUBERNETES_SERVICE_HOST': settings.KUBERNETES_SERVICE_HOST,
    'EXPOSE_TEST_HOOKS': settings.EXPOSE_TEST_HOOKS,
    'SS_AUTH_TOKEN': _auth_token_for_user(getattr(request, "user", None)),
    'chrome_bootstrap': chrome_bootstrap(request),
  }
