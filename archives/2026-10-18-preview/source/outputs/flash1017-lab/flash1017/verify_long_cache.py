"""Check a changed question against the completed million-token prefix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import threading
import time
import urllib.request

from transformers import PreTrainedTokenizerFast
from omlx.patches.deepseek_v41.processing import Processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    cold_path = root/'results/long-context-http-01.json'
    cold = json.loads(cold_path.read_text())['tests'][1]
    assert cold['status'] == 'complete' and cold['passed']
    model = Path(json.loads((root/'results/checkpoint-hashes.json').read_text())['directory'])
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model)
    processor = Processor(tokenizer, None)
    prompt = json.loads((root/'plans/long-context-prompts.json').read_text())['retrieval-1000000']
    encoded = tokenizer.decode(prompt['tokens'], skip_special_tokens=False)
    begin, end = '<｜begin▁of▁sentence｜><｜User｜>', '<｜Assistant｜></think>'
    assert encoded.startswith(begin) and encoded.endswith(end)
    body = encoded[len(begin):-len(end)]
    messages = [{'role': 'system', 'content': f'Document instance {cold_path.stem}-retrieval-1000000. Answer only from the supplied document.'},
                {'role': 'user', 'content': body}]
    original = processor.apply_chat_template(messages, tokenize=True, enable_thinking=False)
    assert hashlib.sha256(json.dumps(original).encode()).hexdigest() == cold['input_sha256']
    document, marker, question = body.rpartition('\nDOCUMENT_END\n')
    assert marker and question
    messages[-1]['content'] = document+marker+'Return only a JSON object containing BIRCH and EMBER with their exact access_code strings from the document.'
    tokens = processor.apply_chat_template(messages, tokenize=True, enable_thinking=False)
    common = next((i for i, pair in enumerate(zip(original, tokens)) if pair[0] != pair[1]), min(len(original), len(tokens)))
    assert common > 990000
    url = 'http://127.0.0.1:8001'
    def cache_stats():
        with urllib.request.urlopen(url+'/admin/api/stats', timeout=5) as response:
            # Never retain the top-level response, which contains local auth settings.
            data = json.load(response)['runtime_cache']
        return next(m for m in data['models'] if m['id'] == model.name)
    before = cache_stats()
    report = {'status': 'running', 'input_tokens': len(tokens), 'common_prefix_tokens': common,
              'input_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
              'scope': 'Existing oMLX prefix caching on a changed question; this is not a new kernel contribution.'}
    assert before['boundary_snapshots']['last_event']['token_count'] > 990000
    previous_id = before['last_prefix_lookup']['request_id']
    done = threading.Event()
    def interrupt(signum, frame):
        raise RuntimeError('Cache reuse failed or the bounded follow-up exceeded 120 seconds')
    signal.signal(signal.SIGUSR1, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    def monitor():
        while not done.wait(.5):
            try:
                lookup = cache_stats().get('last_prefix_lookup', {})
                if lookup.get('request_id') != previous_id and lookup.get('prompt_tokens') == len(tokens):
                    report['cache_lookup'] = lookup
                    if lookup.get('reused_kv_tokens', 0) < 990000:
                        os.kill(os.getpid(), signal.SIGUSR1)
                    return
            except (OSError, ValueError, StopIteration):
                pass
    watcher = threading.Thread(target=monitor, daemon=True)
    started = time.perf_counter()
    text, first, usage = [], None, None
    signal.alarm(120)
    watcher.start()
    try:
        payload = {'model': model.name, 'messages': messages, 'temperature': 0,
                   'max_tokens': 128, 'stream': True, 'stream_options': {'include_usage': True},
                   'chat_template_kwargs': {'enable_thinking': False}}
        request = urllib.request.Request(url+'/v1/chat/completions', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    break
                item = json.loads(data)
                if item.get('error'):
                    raise RuntimeError(item['error'])
                usage = item.get('usage') or usage
                for choice in item.get('choices', []):
                    piece = choice.get('delta', {}).get('content')
                    if piece:
                        first = first if first is not None else time.perf_counter()-started
                        text.append(piece)
        content = ''.join(text)
        match = re.search(r'\{.*?\}', content, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
        correct = {k: parsed.get(k) == cold['expected'][k] for k in ['BIRCH', 'EMBER']}
        cached = (usage or {}).get('prompt_tokens_details', {}).get('cached_tokens', 0)
        report.update(status='complete', passed=all(correct.values()) and cached >= 990000,
                      text=content, correct=correct, usage=usage, cached_tokens=cached)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), text=''.join(text))
        raise
    finally:
        signal.alarm(0)
        done.set(); watcher.join(timeout=6)
        report.update(ttft_seconds=first, total_seconds=time.perf_counter()-started)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
