"""Temporarily prevent the idle original server loading a second huge model."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

RECORD = Path(__file__).with_name('original-memory-reservation.json')
FIELDS = ('memory_guard_tier', 'memory_guard_custom_ceiling_gb')


def call(path, payload=None):
    request = urllib.request.Request('http://127.0.0.1:8000' + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def memory_fields():
    result = call('/admin/api/global-settings')
    memory = result.get('settings', result)['memory']
    return {key: memory[key] for key in FIELDS}


def save(state):
    RECORD.write_text(json.dumps(state, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['reserve', 'restore'])
    args = parser.parse_args()
    if args.action == 'restore':
        state = json.loads(RECORD.read_text())
        current = memory_fields()
        if current != state['original']:
            if current != state['temporary']:
                raise RuntimeError('Settings changed independently; left unchanged')
            call('/admin/api/global-settings', state['original'])
        if memory_fields() != state['original']:
            raise RuntimeError('Original settings not restored')
        state.update(status='restored', restored_at=time.time())
        save(state)
        print('Original memory allowance restored.')
        return
    if RECORD.exists():
        raise FileExistsError('Keep prior reservation evidence; choose a new run folder')
    status = call('/api/status')
    if status.get('active_requests') or status.get('waiting_requests'):
        raise RuntimeError('Original server has active work; did not change it')
    models = status.get('loaded_models', [])
    if any(name != 'Qwen3.8-Flash-Next-oQ8e-mtp' for name in models):
        raise RuntimeError('Unexpected model load; did not change it')
    state = {'original': memory_fields(), 'temporary': {
        'memory_guard_tier': 'custom', 'memory_guard_custom_ceiling_gb': 32},
        'status': 'reserving', 'created_at': time.time(), 'unloaded_models': models}
    save(state)
    call('/admin/api/global-settings', state['temporary'])
    for name in models:
        status = call('/api/status')
        if status.get('active_requests') or status.get('waiting_requests'):
            call('/admin/api/global-settings', state['original'])
            state['status'] = 'restored'; save(state)
            raise RuntimeError('Work arrived; restored memory allowance without unloading')
        call('/admin/api/models/' + name + '/unload', {})
        if name in call('/api/status').get('loaded_models', []):
            raise RuntimeError('Unload failed; inspect durable reservation record')
    if memory_fields() != state['temporary']:
        raise RuntimeError('Temporary memory allowance not applied')
    state['status'] = 'reserved'; save(state)
    print('Idle original model unloaded; temporary memory allowance saved for restoration.')


if __name__ == '__main__':
    main()
