# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MCP stdio server — investigator-first tools over the Scene Metadata HTTP API."""

import json
import urllib.parse

from mcp.server.fastmcp import FastMCP

from scene_metadata.api_client import get_json, persona_allowed, post_json

mcp = FastMCP('scenescape-scene-metadata')


@mcp.tool()
def query_events(
  start: str | None = None,
  end: str | None = None,
  scene_id: str | None = None,
  identity_id: str | None = None,
  region_id: str | None = None,
  tripwire_id: str | None = None,
  event_types: list[str] | None = None,
  limit: int = 200,
) -> str:
  """Query the Timescale event log (investigator, operator, integrator)."""
  if not (
    persona_allowed('investigator')
    or persona_allowed('operator')
    or persona_allowed('integrator')
  ):
    raise RuntimeError('query_events is not enabled for configured MCP personas')
  result = post_json('/v1/events/query', {
    'start': start,
    'end': end,
    'scene_id': scene_id,
    'identity_id': identity_id,
    'region_id': region_id,
    'tripwire_id': tripwire_id,
    'event_types': event_types,
    'limit': limit,
  })
  return json.dumps(result, default=str)


@mcp.tool()
def identity_timeline(
  identity_id: str,
  start: str | None = None,
  end: str | None = None,
  limit: int = 500,
) -> str:
  """Fused track + region + tripwire timeline for one global identity."""
  if not persona_allowed('investigator'):
    raise RuntimeError('identity_timeline requires investigator persona')
  result = get_json(
    f'/v1/identities/{urllib.parse.quote(identity_id, safe="")}/timeline',
    {'start': start, 'end': end, 'limit': str(limit)},
  )
  return json.dumps(result, default=str)


@mcp.tool()
def region_activity(
  region_id: str,
  start: str | None = None,
  end: str | None = None,
  limit: int = 500,
) -> str:
  """Region enter/exit/count/object analytics events in a time window."""
  if not persona_allowed('investigator'):
    raise RuntimeError('region_activity requires investigator persona')
  result = get_json(
    f'/v1/regions/{urllib.parse.quote(region_id, safe="")}/activity',
    {'start': start, 'end': end, 'limit': str(limit)},
  )
  return json.dumps(result, default=str)


@mcp.tool()
def tripwire_crossings(
  tripwire_id: str,
  start: str | None = None,
  end: str | None = None,
  limit: int = 500,
) -> str:
  """Ordered tripwire crossing events."""
  if not persona_allowed('investigator'):
    raise RuntimeError('tripwire_crossings requires investigator persona')
  result = get_json(
    f'/v1/tripwires/{urllib.parse.quote(tripwire_id, safe="")}/crossings',
    {'start': start, 'end': end, 'limit': str(limit)},
  )
  return json.dumps(result, default=str)


@mcp.tool()
def get_system_status() -> str:
  """Operator view: health and ingest lag stats."""
  if not persona_allowed('operator'):
    raise RuntimeError('get_system_status requires operator persona')
  return json.dumps(get_json('/v1/health'), default=str)


def main() -> None:
  mcp.run(transport='stdio')


if __name__ == '__main__':
  main()
