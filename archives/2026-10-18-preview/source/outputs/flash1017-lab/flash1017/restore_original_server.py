"""Restore only the temporary memory fields this experiment changed."""
import json
from pathlib import Path
import urllib.request


def main():
    root=Path(__file__).resolve().parents[1]
    record=root/'server-state/original-server-memory-reservation.json'
    if not record.exists():return
    state=json.loads(record.read_text())
    if state['status']=='restored':return
    def call(payload=None):
        request=urllib.request.Request('http://127.0.0.1:8000/admin/api/global-settings',
            data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)
    settings=call()
    memory=settings.get('settings',settings)['memory']
    current={k:memory[k] for k in state['original']}
    if current==state['original']:
        state['status']='restored'
    elif current!=state['temporary']:
        raise RuntimeError('The original server settings changed independently; they were left unchanged')
    else:
        result=call(state['original'])
        if not result.get('success'):raise RuntimeError(result)
        check=call()
        memory=check.get('settings',check)['memory']
        if any(memory[k]!=v for k,v in state['original'].items()):
            raise RuntimeError('Original settings were not restored')
        state['status']='restored'
    record.write_text(json.dumps(state,indent=2)+'\n')
    print('Original oMLX memory allowance restored.')


if __name__=='__main__':main()
