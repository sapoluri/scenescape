# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared mesh generate/status HTTP handlers for Django views and Token API."""

import traceback

from django.db import transaction

from scene_common import log


def start_mesh_generation_payload(scene, mesh_type="mesh", uploaded_map=None):
  """Run mesh generation start. Returns (payload_dict, http_status)."""
  try:
    from manager.mesh_generator import MeshGenerator

    mesh_generator = MeshGenerator()
    result = mesh_generator.startMeshGeneration(
      scene, mesh_type, uploaded_map=uploaded_map
    )
    if result.get("success"):
      return {
        "success": True,
        "message": "Mesh generated successfully",
        "request_id": result["request_id"],
        "processing_time": result.get("processing_time", 0),
      }, 200

    return {
      "success": False,
      "error": result.get(
        "error", "Unknown error occurred while generating mesh"
      ),
      "processing_time": result.get("processing_time", 0),
    }, 400
  except Exception as e:
    log.error(f"Mesh generation error: {e}")
    log.error(f"Traceback: {traceback.format_exc()}")
    return {
      "success": False,
      "error": "An internal error occurred while generating mesh",
    }, 500


def mesh_generation_status_payload(scene, request_id):
  """Poll/finalize mesh status. Returns (payload_dict, http_status)."""
  if not request_id:
    return {"success": False, "error": "missing request_id"}, 400

  try:
    from manager.mesh_generator import MeshGenerator

    mesh_generator = MeshGenerator()
    status_data = mesh_generator.mapping_client.getReconstructionStatus(
      request_id
    )

    if not status_data.get("success"):
      return status_data, 200

    state = status_data.get("state")
    if state != "complete":
      return status_data, 200

    with transaction.atomic():
      from manager.models import Scene

      scene = Scene.objects.select_for_update().get(pk=scene.pk)

      if hasattr(scene, "mesh_state") and scene.mesh_state == "complete":
        status_data["finalized"] = True
        return status_data, 200

      finalize_result = mesh_generator.finalizeMeshFromStatus(
        scene, request_id
      )

      if not finalize_result.get("success"):
        if hasattr(scene, "mesh_state"):
          scene.mesh_state = "failed"
          scene.save(update_fields=["mesh_state"])
        return finalize_result, 500

      if hasattr(scene, "mesh_state"):
        scene.mesh_state = "complete"
        scene.save(update_fields=["mesh_state"])

    status_data["finalized"] = True
    if finalize_result.get("unanchored_cameras"):
      status_data["unanchored_cameras"] = finalize_result["unanchored_cameras"]
    return status_data, 200

  except Exception as e:
    log.error(f"Mesh status error: {e}")
    log.error(f"Traceback: {traceback.format_exc()}")
    return {
      "success": False,
      "error": "An internal error occurred while getting mesh status",
    }, 500
