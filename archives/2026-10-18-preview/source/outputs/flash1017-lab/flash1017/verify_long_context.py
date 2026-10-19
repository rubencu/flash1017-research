"""Run frozen retrieval probes through the real streaming HTTP serving path."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import urllib.request

import psutil
from transformers import PreTrainedTokenizerFast
from omlx.patches.deepseek_v41.processing import Processor
from omlx.utils.proc_memory import get_phys_footprint, _RusageInfoV4, _proc_pid_rusage, _RUSAGE_INFO_V4


def io_for(pid):
    info = _RusageInfoV4()
    if _proc_pid_rusage(pid, _RUSAGE_INFO_V4, ctypes.byref(info)):
        return {}
    return {'disk_read_bytes': int(info.ri_diskio_bytesread), 'disk_write_bytes': int(info.ri_diskio_byteswritten)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--url',default='http://127.0.0.1:8001')
    parser.add_argument('--only', choices=['retrieval-131072', 'retrieval-1000000'])
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1]
    model=Path(json.loads((root/'results/checkpoint-hashes.json').read_text())['directory'])
    stamp=json.loads((root/'server-state/server-process.json').read_text())
    server=psutil.Process(stamp['pid'])
    if server.create_time()!=stamp['created']:raise RuntimeError('Server PID changed')
    tokenizer=PreTrainedTokenizerFast.from_pretrained(model)
    processor=Processor(tokenizer,None)
    prompts=json.loads((root/'plans/long-context-prompts.json').read_text())
    expected=json.loads((root/'fixtures/long-context-answers.json').read_text())
    report={'server':stamp,'tests':[], 'scope':'Synthetic five-key retrieval through the local HTTP API, not a general long-context quality benchmark.'}
    def save():args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    for name,prompt in prompts.items():
        if args.only and name != args.only:
            continue
        encoded=tokenizer.decode(prompt['tokens'],skip_special_tokens=False)
        begin='<｜begin▁of▁sentence｜><｜User｜>'
        end='<｜Assistant｜></think>'
        if not encoded.startswith(begin) or not encoded.endswith(end):
            raise ValueError('Unexpected frozen chat framing')
        body=encoded[len(begin):-len(end)]
        # A distinct early system message prevents reuse of the preceding test's
        # prefix. Record actual tokenization; API framing adds a few tokens.
        messages=[{'role':'system','content':f'Document instance {args.output.stem}-{name}. Answer only from the supplied document.'}, {'role':'user','content':body}]
        actual_tokens=processor.apply_chat_template(messages,tokenize=True,enable_thinking=False)
        record={'name':name,'status':'running','actual_input_tokens':len(actual_tokens),
                'input_sha256':hashlib.sha256(json.dumps(actual_tokens).encode()).hexdigest(),
                'user_text_sha256':hashlib.sha256(body.encode()).hexdigest(),
                'expected':expected[name]['expected'], 'needle_positions_in_original':expected[name]['needle_token_positions']}
        report['tests'].append(record);save()
        done=threading.Event();samples=[]
        def snapshot():
            vm=psutil.virtual_memory()
            try:
                footprint=get_phys_footprint(server.pid)
            except (psutil.NoSuchProcess, ProcessLookupError):
                footprint=0
            samples.append({'footprint':footprint,'available':vm.available,'swap':psutil.swap_memory().used})
        def sample():
            while not done.wait(.5):
                snapshot()
        snapshot()
        thread=threading.Thread(target=sample,daemon=True);thread.start()
        io_start=io_for(server.pid)
        started=time.perf_counter();first=None;content=[];usage=None;finish=None
        try:
            payload={'model':model.name,'messages':messages,'temperature':0,'max_tokens':256,
                     'stream':True,'stream_options':{'include_usage':True},
                     'chat_template_kwargs':{'enable_thinking':False}}
            request=urllib.request.Request(args.url+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=3600) as response:
                for line in response:
                    if not line.startswith(b'data: '):continue
                    data=line[6:].strip()
                    if data==b'[DONE]':break
                    item=json.loads(data)
                    if item.get('error'):raise RuntimeError(item['error'])
                    if item.get('usage'):usage=item['usage']
                    for choice in item.get('choices',[]):
                        piece=choice.get('delta',{}).get('content')
                        if piece:
                            if first is None:first=time.perf_counter()-started
                            content.append(piece)
                        finish=choice.get('finish_reason') or finish
            text=''.join(content)
            match=re.search(r'\{.*?\}',text,re.DOTALL)
            parsed=json.loads(match.group()) if match else {}
            correct={key:parsed.get(key)==value for key,value in record['expected'].items()}
            record.update(status='complete',text=text,parsed=parsed,correct=correct,passed=all(correct.values()),usage=usage,finish_reason=finish)
        except BaseException as exc:
            record.update(status='failed',error=repr(exc),text=''.join(content))
            raise
        finally:
            done.set();thread.join()
            io_end=io_for(server.pid)
            record.update(total_seconds=time.perf_counter()-started,ttft_seconds=first,
              resources={'sampling_interval_seconds':.5,'peak_process_footprint_bytes':max(s['footprint'] for s in samples),
                         'minimum_system_available_bytes':min(s['available'] for s in samples),
                         'swap_start_bytes':samples[0]['swap'],'swap_peak_bytes':max(s['swap'] for s in samples),
                         **{k+'_delta':v-io_start.get(k,v) for k,v in io_end.items()}})
            save()
            print(json.dumps({k:record.get(k) for k in ['name','status','passed','actual_input_tokens','ttft_seconds','total_seconds']}),flush=True)


if __name__=='__main__':main()
