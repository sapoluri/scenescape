# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download PhysicalAI Lab_000 and check attribute log, graph, and MCP queries.

The dataset is not stored in git. A machine with internet access can fetch the
Lab_000 calibration, map, and ground truth, load them into Timescale, and run
the same MCP tools used by an investigator. RGB videos are not required.

Ground truth has no clothing or gender labels. The loader stores recomputable
track facts in the classificationPolicy metadata shape, then projects one
HAS_ATTRIBUTE edge per key.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import threading
import time
import urllib.request

from scene_metadata.db import Database
from scene_metadata.physicalai_loader import (
  REGION_ID,
  REGION_Y,
  SCENE_ID,
  TRIPWIRE_X,
  load_directory,
)
from scene_metadata.query_app import create_app
from scene_metadata.store import EventStore

DATASET_PAGE = 'https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces'
DATASET_URL = (
  'https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/resolve/main/'
  'MTMC_Tracking_2025/val/Lab_000'
)
DATASET_FILES = ('calibration.json', 'ground_truth.json', 'map.png')
DEFAULT_DEPLOYMENT = 'physicalai-lab-attributes'


def default_data_dir() -> Path:
  """Directory for the downloaded Lab_000 files. Git ignores the default path."""
  override = os.environ.get('PHYSICALAI_LAB_DIR')
  if override:
    return Path(override)
  return Path('tools/reid_evaluation/output/physicalai-lab-data')


def download_scene(dest: Path, refresh: bool = False) -> None:
  """Fetch Lab_000 annotations from Hugging Face. Skip files already present."""
  dest.mkdir(parents=True, exist_ok=True)
  for name in DATASET_FILES:
    target = dest / name
    if target.is_file() and target.stat().st_size > 0 and not refresh:
      continue
    url = f'{DATASET_URL}/{name}?download=true'
    request = urllib.request.Request(url, headers={'User-Agent': 'scenescape-scene-metadata'})
    partial = target.with_suffix(target.suffix + '.partial')
    try:
      with urllib.request.urlopen(request, timeout=180) as response, partial.open('wb') as handle:
        while True:
          chunk = response.read(1024 * 1024)
          if not chunk:
            break
          handle.write(chunk)
      partial.replace(target)
    finally:
      if partial.exists():
        partial.unlink()


def expected_labels(ground_truth: dict) -> dict[str, dict[str, str]]:
  """Attribute labels recomputed from the dataset, not from the loader."""
  by_id: dict[int, list[tuple[int, dict]]] = defaultdict(list)
  for frame in sorted(int(key) for key in ground_truth):
    for obj in ground_truth[str(frame)]:
      if obj.get('object type') != 'Person':
        continue
      by_id[obj['object id']].append((frame, obj))
  return {f'person-{object_id}': _labels(samples) for object_id, samples in by_id.items()}


def _labels(samples: list[tuple[int, dict]]) -> dict[str, str]:
  entered = 0
  crossings = 0
  in_region = False
  prev_x = None
  first_enter = None
  seen: set[str] = set()
  linked_before: set[str] = set()
  for _frame, obj in samples:
    loc = obj.get('3d location') or [None, None, None]
    x_pos, y_pos = loc[0], loc[1]
    visible = set(obj.get('2d bounding box visible') or {})
    new_cameras = visible - seen
    inside = y_pos is not None and y_pos > REGION_Y
    if inside and not in_region:
      entered += 1
      if first_enter is None:
        first_enter = _frame
    if first_enter is None:
      linked_before |= new_cameras
    seen |= visible
    if prev_x is not None and x_pos is not None and (prev_x < TRIPWIRE_X) != (x_pos < TRIPWIRE_X):
      crossings += 1
    prev_x = x_pos
    in_region = inside
  return {
    'north_entries': str(entered),
    'tripwire_crossings': str(crossings),
    'cameras_before_north': str(len(linked_before)),
  }


def pin_frame_zero(ground_truth: dict, expected: dict[str, dict[str, str]]) -> None:
  """person-0 is inside the north region on the first labeled frame."""
  first = next(obj for obj in ground_truth['0'] if obj.get('object id') == 0)
  if first['3d location'][1] <= REGION_Y:
    raise AssertionError('Lab_000 person-0 is expected inside y > 0 at frame 0')
  if expected['person-0']['cameras_before_north'] != '0':
    raise AssertionError('no camera can be linked before an entry that happens at frame 0')


def _who(expected: dict[str, dict[str, str]], name: str, label: str) -> list[str]:
  return sorted(identity for identity, labels in expected.items() if labels.get(name) == label)


def _who_pair(expected: dict[str, dict[str, str]], entries: str, crossings: str) -> list[str]:
  return sorted(
    identity for identity, labels in expected.items()
    if labels.get('north_entries') == entries and labels.get('tripwire_crossings') == crossings
  )


def assert_attribute_results(fetch_events, fetch_attributes, expected, ground_truth) -> list[str]:
  """Check log queries and projected attributes. Return lines for the demo log."""
  pin_frame_zero(ground_truth, expected)
  lines = []
  names = sorted({name for labels in expected.values() for name in labels})
  for name in names:
    labels = sorted({item[name] for item in expected.values()})
    if len(labels) < 2:
      raise AssertionError(f'{name} does not distinguish Lab_000 people: {labels}')
    for label in labels:
      for event_type in ('attribute_observed', 'track_started'):
        got = fetch_events({name: label}, [event_type])
        want = _who(expected, name, label)
        if got != want:
          raise AssertionError(f'{event_type} {name}={label}: got {got}, want {want}')
        lines.append(f'{event_type} {name}={label} -> {got}')
    missing = fetch_events({name: 'not-in-the-dataset'}, ['attribute_observed'])
    if missing:
      raise AssertionError(f'{name}=not-in-the-dataset returned {missing}')
    lines.append(f'{name}=not-in-the-dataset -> []')

  pair_owners = None
  pair_labels = None
  for labels in expected.values():
    owners = _who_pair(expected, labels['north_entries'], labels['tripwire_crossings'])
    if len(owners) == 1:
      pair_owners = owners
      pair_labels = labels
      break
  if pair_owners is None or pair_labels is None:
    raise AssertionError('north_entries and tripwire_crossings do not isolate one person')
  combined = fetch_events(
    {
      'north_entries': pair_labels['north_entries'],
      'tripwire_crossings': pair_labels['tripwire_crossings'],
    },
    ['attribute_observed'],
  )
  if combined != pair_owners:
    raise AssertionError(f'combined filter: got {combined}, want {pair_owners}')
  lines.append(
    'north_entries='
    f"{pair_labels['north_entries']} and tripwire_crossings="
    f"{pair_labels['tripwire_crossings']} -> {combined}"
  )

  before_label = next(
    label for label in sorted({item['cameras_before_north'] for item in expected.values()})
    if len(_who(expected, 'cameras_before_north', label)) == 1
  )
  entered = fetch_events({'cameras_before_north': before_label}, ['region_enter'])
  if entered != _who(expected, 'cameras_before_north', before_label):
    raise AssertionError(
      f'region_enter with cameras_before_north={before_label} returned {entered}')
  lines.append(f'region {REGION_ID} enter and cameras_before_north={before_label} -> {entered}')

  for identity, labels in sorted(expected.items()):
    projected = fetch_attributes(identity)
    if projected != labels:
      raise AssertionError(f'projected attributes for {identity}: got {projected}, want {labels}')
    lines.append(f'graph {identity} -> {projected}')
  return lines


def replace_scene(db: Database, deployment_id: str, data_dir: Path) -> int:
  """Replace one deployment with a fresh load of the dataset directory."""
  store = EventStore(db)
  with db.connection() as conn:
    conn.execute(
      'DELETE FROM scenescape_graph_edges WHERE deployment_id = %s', (deployment_id,))
    conn.execute(
      'DELETE FROM scenescape_events WHERE deployment_id = %s', (deployment_id,))
    conn.commit()
  return store.insert_events(load_directory(data_dir, deployment_id))


def _wait_for_server(server, timeout: float) -> None:
  deadline = time.time() + timeout
  while not getattr(server, 'started', False):
    if time.time() > deadline:
      raise RuntimeError('scene metadata HTTP server did not start')
    time.sleep(0.05)


def serve_and_check(db: Database, deployment_id: str, ground_truth: dict, port: int) -> list[str]:
  """Start the HTTP API and call the MCP tool functions against it."""
  import uvicorn

  from scene_metadata.mcp_server import identity_attributes, query_events

  os.environ['SCENE_METADATA_DEPLOYMENT_ID'] = deployment_id
  os.environ['SCENE_METADATA_API_URL'] = f'http://127.0.0.1:{port}'
  config = uvicorn.Config(
    create_app(db), host='127.0.0.1', port=port, log_level='warning')
  server = uvicorn.Server(config)
  thread = threading.Thread(target=server.run, name='scene-metadata-demo', daemon=True)
  thread.start()
  try:
    _wait_for_server(server, 20)
    expected = expected_labels(ground_truth)

    def fetch_events(attributes, event_types=None):
      payload = json.loads(query_events(
        scene_id=SCENE_ID,
        attributes=attributes,
        event_types=event_types,
        limit=50,
      ))
      return sorted({row['identity_id'] for row in payload['events']})

    def fetch_attributes(identity):
      payload = json.loads(identity_attributes(identity))
      return {item['name']: item['label'] for item in payload['attributes']}

    return assert_attribute_results(fetch_events, fetch_attributes, expected, ground_truth)
  finally:
    server.should_exit = True
    thread.join(timeout=5)


def main(argv: list[str] | None = None) -> int:
  """Download Lab_000 if needed, load it, and check MCP attribute queries."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data-dir', type=Path, default=None)
  parser.add_argument('--refresh', action='store_true', help='Download the files again')
  parser.add_argument('--deployment-id', default=DEFAULT_DEPLOYMENT)
  parser.add_argument('--port', type=int, default=8765)
  args = parser.parse_args(argv)
  data_dir = args.data_dir or default_data_dir()
  download_scene(data_dir, refresh=args.refresh)
  print(f'Dataset directory: {data_dir}')
  print(f'Source: {DATASET_PAGE} (CC-BY-4.0). Files are not part of this repository.')
  ground_truth = json.loads((data_dir / 'ground_truth.json').read_text(encoding='utf-8'))
  db = Database()
  db.open()
  try:
    inserted = replace_scene(db, args.deployment_id, data_dir)
    print(f'Inserted {inserted} events into deployment {args.deployment_id}')
    for line in serve_and_check(db, args.deployment_id, ground_truth, args.port):
      print(line)
  finally:
    db.close()
  print('Attribute log, graph projection, and MCP queries match Lab_000 ground truth.')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
