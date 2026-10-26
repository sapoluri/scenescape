# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MCP stdio server — investigator-first tools over the Scene Metadata HTTP API."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from mcp.server.fastmcp import FastMCP

API_BASE = os.environ.get('SCENE_METADATA_API_URL', 'http://127.0.0.1:8080').rstrip('/')
PERSONAS = {
  p.strip()
  for p in os.environ.get(
    'SCENE_METADATA_MCP_PERSONAS', 'investigator,operator,integrator').split(',')
  if p.strip()
}

mcp = FastMCP('scenescape-scene-metadata')


def _get(path: str, query: dict | None = None) -> dict:
  url = API_BASE + path
  if query:
    params = urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
    if params:
      url = f'{url}?{params}'
  request = urllib.request.Request(url, headers={'Accept': 'application/json'})
  try:
    with urllib.request.urlopen(request, timeout=30) as response:
      return json.loads(response.read().decode('utf-8'))
  except urllib.error.HTTPError as err:
    body = err.read().decode('utf-8', errors='replace')
    raise RuntimeError(f'API {path} failed ({err.code}): {body}') from err


def _post(path: str, payload: dict) -> dict:
  data = json.dumps(payload).encode('utf-8')
  request = urllib.request.Request(
    API_BASE + path,
    data=data,
    headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
    method='POST',
  )
  with urllib.request.urlopen(request, timeout=30) as response:
    return json.loads(response.read().decode('utf-8'))


def _allowed(persona: str) -> bool:
  return persona in PERSONAS


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
  if not (_allowed('investigator') or _allowed('operator') or _allowed('integrator')):
    raise RuntimeError('query_events is not enabled for configured MCP personas')
  result = _post('/v1/events/query', {
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
  if not _allowed('investigator'):
    raise RuntimeError('identity_timeline requires investigator persona')
  result = _get(
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
  if not _allowed('investigator'):
    raise RuntimeError('region_activity requires investigator persona')
  result = _get(
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
  if not _allowed('investigator'):
    raise RuntimeError('tripwire_crossings requires investigator persona')
  result = _get(
    f'/v1/tripwires/{urllib.parse.quote(tripwire_id, safe="")}/crossings',
    {'start': start, 'end': end, 'limit': str(limit)},
  )
  return json.dumps(result, default=str)


@mcp.tool()
def get_system_status() -> str:
  """Operator view: health and ingest lag stats."""
  if not _allowed('operator'):
    raise RuntimeError('get_system_status requires operator persona')
  return json.dumps(_get('/v1/health'), default=str)


def main() -> None:
  mcp.run(transport='stdio')


if __name__ == '__main__':
  main()
