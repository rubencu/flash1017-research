"""Exercise the launchers while a real response is streaming, then recover."""
import argparse
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    model = Path(json.loads((root/'results/checkpoint-hashes.json').read_text())['directory']).name
    url = 'http://127.0.0.1:8001'
    report = {'status': 'running', 'stream': {}}
    started = threading.Event()

    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')

    def request(payload):
        return urllib.request.Request(url+'/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})

    def stream():
        try:
            payload = {'model': model, 'messages': [{'role': 'user', 'content':
                'Write a long, detailed engineering guide to transaction logs and database recovery.'}],
                'temperature': 0, 'max_tokens': 4096, 'stream': True,
                'chat_template_kwargs': {'enable_thinking': False}}
            with urllib.request.urlopen(request(payload), timeout=120) as response:
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    data = line[6:].strip()
                    if data == b'[DONE]':
                        break
                    item = json.loads(data)
                    if item.get('error'):
                        report['stream']['server_error'] = item['error']
                    if any(c.get('delta', {}).get('content') for c in item.get('choices', [])):
                        started.set()
        except Exception as exc:
            report['stream']['ended_with'] = repr(exc)
        finally:
            report['stream']['closed'] = True

    save()
    thread = threading.Thread(target=stream, daemon=True)
    thread.start()
    try:
        if not started.wait(30):
            raise TimeoutError('No visible text before stop test')
        report['stream']['visible_text_before_stop'] = True
        before = time.perf_counter()
        stopped = subprocess.run([str(root/'run.sh'), '-m', 'flash1017.stop'],
                                 capture_output=True, text=True, timeout=90)
        report['stop'] = {'seconds': time.perf_counter()-before,
                          'returncode': stopped.returncode, 'output': stopped.stdout,
                          'stderr': stopped.stderr}
        if stopped.returncode:
            raise RuntimeError('Stop launcher did not complete')
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError('Streaming connection survived server stop')
        save()
        print('Stop during generation passed. Restarting the model.', flush=True)
        before = time.perf_counter()
        launched = subprocess.run([str(root/'run.sh'), '-u', '-m', 'flash1017.launch', '--no-browser'],
                                 capture_output=True, text=True, timeout=360)
        report['restart'] = {'seconds': time.perf_counter()-before,
                             'returncode': launched.returncode, 'output': launched.stdout,
                             'stderr': launched.stderr}
        if launched.returncode:
            raise RuntimeError('Restart launcher did not complete')
        payload = {'model': model, 'messages': [{'role': 'user', 'content': 'Reply exactly: RESTART1017_OK'}],
                   'temperature': 0, 'max_tokens': 32,
                   'chat_template_kwargs': {'enable_thinking': False}}
        with urllib.request.urlopen(request(payload), timeout=120) as response:
            result = json.load(response)
        report['recovery'] = result
        report['passed'] = result['choices'][0]['message'].get('content', '').strip() == 'RESTART1017_OK'
        if not report['passed']:
            raise RuntimeError('Restarted model failed the recovery prompt')
        report['status'] = 'complete'
        print('Restart and recovery passed.', flush=True)
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
