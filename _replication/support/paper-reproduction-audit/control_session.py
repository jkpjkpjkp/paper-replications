#!/usr/bin/env python3
"""Foreground supervisor control client for the already-running local daemon."""
import datetime
import json
from pathlib import Path
import sys
from local_control import LocalControl

root = Path(__file__).resolve().parent
receipts = root / 'control-receipts'
receipts.mkdir(exist_ok=True)
client = LocalControl()
print('CONTROL_READY existing daemon, no restart or new thread', flush=True)
try:
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if request.get('method') == 'close':
            break
        method = request['method']
        if method not in {'thread/read', 'thread/turns/list', 'thread/items/list',
                          'thread/unarchive', 'thread/resume', 'turn/start', 'turn/steer',
                          'thread/archive', 'thread/loaded/list'}:
            print(json.dumps({'error': 'Method not allowed by supervisor client', 'method': method}), flush=True)
            continue
        result = client.call(method, request.get('params', {}))
        body = result.get('result', {})
        thread = body.get('thread', {})
        turn = body.get('turn', {})
        summary = {'tag': request.get('tag'), 'method': method,
                   'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'error': result.get('error'), 'result_keys': list(body),
                   'thread': {k: thread.get(k) for k in ['id', 'status', 'path', 'cwd'] if k in thread},
                   'turn': {k: turn.get(k) for k in ['id', 'status', 'error'] if k in turn}}
        if method in {'thread/turns/list', 'thread/items/list', 'thread/loaded/list'}:
            summary['result'] = body
        if method in {'thread/unarchive', 'thread/resume', 'turn/start', 'turn/steer', 'thread/archive'}:
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            receipt = {'request': request, 'response_summary': summary}
            (receipts / f'{stamp}.json').write_text(json.dumps(receipt, indent=2) + '\n')
        events = [v for v in client.events if v.get('method') in
                  {'turn/completed', 'thread/status/changed', 'item/completed'}]
        summary['event_count'] = len(client.events)
        summary['completed_turns'] = [v['params'] for v in events if v['method'] == 'turn/completed']
        client.events.clear()
        print(json.dumps(summary), flush=True)
finally:
    client.close()
