# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""HTTP query API for investigator, operator, and integrator clients."""

import os
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from scene_metadata.db import Database
from scene_metadata.store import EventStore


def _deployment_id() -> str:
  return os.environ.get('SCENE_METADATA_DEPLOYMENT_ID', 'default')


def create_app(database: Database | None = None) -> FastAPI:
  db = database or Database()
  db.open()
  store = EventStore(db)
  app = FastAPI(title='Scenescape Scene Metadata', version='0.1.0')

  class EventQuery(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None
    scene_id: Optional[str] = None
    identity_id: Optional[str] = None
    region_id: Optional[str] = None
    tripwire_id: Optional[str] = None
    event_types: Optional[list[str]] = None
    limit: int = Field(default=200, ge=1, le=2000)

  @app.get('/v1/health')
  def health() -> dict[str, Any]:
    dep = _deployment_id()
    stats = store.ingest_stats(dep)
    return {
      'status': 'ok',
      'component': 'scene-metadata',
      'deployment_id': dep,
      'ingest': stats,
    }

  @app.post('/v1/events/query')
  def query_events(body: EventQuery) -> dict[str, Any]:
    events = store.query_events(
      deployment_id=_deployment_id(),
      start=body.start,
      end=body.end,
      scene_id=body.scene_id,
      identity_id=body.identity_id,
      region_id=body.region_id,
      tripwire_id=body.tripwire_id,
      event_types=body.event_types,
      limit=body.limit,
    )
    return {'events': events, 'count': len(events)}

  @app.get('/v1/identities/{identity_id}/timeline')
  def identity_timeline(
    identity_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 500,
  ) -> dict[str, Any]:
    events = store.identity_timeline(
      deployment_id=_deployment_id(),
      identity_id=identity_id,
      start=start,
      end=end,
      limit=limit,
    )
    return {'identity_id': identity_id, 'events': events, 'count': len(events)}

  @app.get('/v1/regions/{region_id}/activity')
  def region_activity(
    region_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 500,
  ) -> dict[str, Any]:
    events = store.query_events(
      deployment_id=_deployment_id(),
      region_id=region_id,
      start=start,
      end=end,
      event_types=['region_enter', 'region_exit', 'region_count', 'region_objects'],
      limit=limit,
    )
    return {'region_id': region_id, 'events': events, 'count': len(events)}

  @app.get('/v1/tripwires/{tripwire_id}/crossings')
  def tripwire_crossings(
    tripwire_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 500,
  ) -> dict[str, Any]:
    events = store.query_events(
      deployment_id=_deployment_id(),
      tripwire_id=tripwire_id,
      start=start,
      end=end,
      event_types=['tripwire_cross'],
      limit=limit,
    )
    return {'tripwire_id': tripwire_id, 'events': events, 'count': len(events)}

  @app.get('/v1/graph/identities/{identity_id}/regions')
  def graph_identity_regions(identity_id: str) -> dict[str, Any]:
    dep = _deployment_id()
    with db.connection() as conn:
      rows = conn.execute(
        """
        SELECT relation, dst_id AS region_id, scene_id, valid_from, valid_to, payload
        FROM scenescape_graph_edges
        WHERE deployment_id = %s
          AND src_kind = 'identity' AND src_id = %s
          AND dst_kind = 'region'
        ORDER BY valid_from ASC
        LIMIT 500
        """,
        (dep, identity_id),
      )
    return {'identity_id': identity_id, 'edges': list(rows)}

  @app.get('/v1/graph/identities/{identity_id}/tripwires')
  def graph_identity_tripwires(identity_id: str) -> dict[str, Any]:
    dep = _deployment_id()
    with db.connection() as conn:
      rows = conn.execute(
        """
        SELECT relation, dst_id AS tripwire_id, scene_id, valid_from, valid_to, payload
        FROM scenescape_graph_edges
        WHERE deployment_id = %s
          AND src_kind = 'identity' AND src_id = %s
          AND dst_kind = 'tripwire'
        ORDER BY valid_from ASC
        LIMIT 500
        """,
        (dep, identity_id),
      )
    return {'identity_id': identity_id, 'edges': list(rows)}

  @app.on_event('shutdown')
  def _shutdown():
    db.close()

  return app
