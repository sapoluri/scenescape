# SPDX-FileCopyrightText: (C) 2021 - 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist

def _auth_token_for_user(user):
  if user is None or not getattr(user, "is_authenticated", False):
    return ""
  try:
    token = user.auth_token
    return str(token) if token else ""
  except ObjectDoesNotExist:
    return ""

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
  }
