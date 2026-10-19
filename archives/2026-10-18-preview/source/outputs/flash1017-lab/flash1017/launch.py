"""Start the qualified local server and open its chat page when it is ready."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import webbrowser

import psutil


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    state=root/'server-state';state.mkdir(exist_ok=True)
    stamp=state/'server-process.json'
    url='http://127.0.0.1:8001'

    def owned_process():
        if not stamp.exists():return None
        try:
            info=json.loads(stamp.read_text())
            process=psutil.Process(info['pid'])
            return process if process.create_time()==info['created'] else None
        except (psutil.NoSuchProcess,ValueError,KeyError):return None

    def ready():
        if owned_process() is None:return False
        try:
            with urllib.request.urlopen(url+'/health',timeout=2) as response:
                data=json.load(response)
            return (data.get('engine_pool') or {}).get('loaded_count',0)==1
        except (OSError,ValueError):return False

    child=None
    log=state/'server.log'
    if owned_process() is None:
        with log.open('a') as output:
            child=subprocess.Popen([str(root/'run.sh'),'-u','-m','flash1017.serve'],
                cwd=root,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True)
    print('Loading DeepSeek V4.1 Flash 1017. The first load usually takes about half a minute.',flush=True)
    deadline=time.monotonic()+300
    while not ready():
        if child is not None and child.poll() is not None:
            tail='\n'.join(log.read_text(errors='replace').splitlines()[-12:])
            raise RuntimeError('The server could not start.\n'+tail)
        if time.monotonic()>deadline:
            raise TimeoutError(f'Server is still loading. Inspect {log}')
        time.sleep(1)
    chat=url+'/admin/chat'
    print('Ready: '+chat,flush=True)
    if not args.no_browser:webbrowser.open(chat)


if __name__=='__main__':main()
