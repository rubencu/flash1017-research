"""Synthetic evidence tests only; never imports inference modules or uses GPU."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import analyze_performance_v2 as a


def fixture():
    runtime={"ced_prefill":False,"prefill_step":8192,"engram_prefetch_bytes":64*2**20,"mtp":True,"depth":5}
    profile={"qualified":True,"runtime":runtime,"packages":{"mlx":"fixture"},
             "inference_files_sha256":{"_runtime/omlx/test.py":"native-sha","flash1017/bench.py":"b"*64}}
    plan={"design":{"suite":"performance","measured_pairs":12,"measured_jobs":24,"warmup_jobs":2,"generation_tokens":128},
          "profiles":{arm:copy.deepcopy(runtime) for arm in a.ARMS},"prompts":{},"jobs":[],
          "fixture_manifest":"fixture-placeholder","fixture_manifest_sha256":"pending","source_manifest_sha256":"pending"}
    entries=[("warmup",16384)] + [(source,n) for n in (16384,32768) for source in ("cpython","curl","moby-dick","pride-prejudice")]
    entries += [("cpython",65536),("curl",65536),("pride-prejudice",65536),("moby-dick",131072)]
    for i,(source,n) in enumerate(entries):
        name=f"{source}-{n}"
        ids=[i+1]*n
        prompt={"id":name,"prompt_tokens":n,"tokens":ids,"input_tokens_sha256":a.digest(ids),"max_tokens":128,
                "force_length":True,"temperature":1.0,"top_p":1.0}
        if i:prompt["source"]=source
        plan["prompts"][name]=prompt
        order=a.ARMS if i==0 or i%2 else a.ARMS[::-1]
        for arm in order:
            plan["jobs"].append({"profile":arm,"prompt":name,"warmup":i==0,"repeat":0,"seed":1017+i})
    rows={"status":"active","enabled":False,"per_table_budget_bytes":a.TABLE_BUDGET,
          "max_retained_bytes":a.TOTAL_BUDGET,"retained_bytes":0,"retired_tables":[],
          "live_tables":[{"cache_id":i,"table":f"layer{i}.weight","budget":a.TABLE_BUDGET,"nbytes":0,"closed":False,
                          "resident":False,"cache_disabled_after_allocation_failure":False,"counts":{},
                          "unique_hits":0,"unique_queries":0,"admitted":0} for i in range(2)]}
    q={"status":"complete_validated","mode":"performance","model_cleanup_completed":True,"instance_hooks_restored":True,
       "source_hashes_reverified":True,"mismatch_pending":False,"worker_threads":[10,11],"loading_thread":12,
       "started_at_unix":100,"completed_at_unix":200,"pairs":[],"installation":{"rows":copy.deepcopy(rows)}}
    records=[{"type":"manifest","plan":plan,"packages":profile["packages"],"runtime_files_sha256":{"test.py":"native-sha"},
              "experiment_files_sha256":{"bench.py":"b"*64},"model":"/fixture/model"},{"type":"load","seconds":1}]
    for i,job in enumerate(plan["jobs"]):
        p=plan["prompts"][job["prompt"]]; enabled=job["profile"]=="fastpath64"
        before=copy.deepcopy(rows);rows["enabled"]=enabled
        for t in rows["live_tables"]:
            c=t["counts"]
            def inc(k,n=1): c[k]=c.get(k,0)+n
            inc("calls");inc("native_read_calls")
            if enabled:
                missing=2 if not t["nbytes"] else 1
                t["nbytes"]=1024;t["unique_queries"]+=2;t["unique_hits"]+=2-missing;t["admitted"]+=missing
                for k,n in (("cache_calls",1),("requested_rows",3),("unique_rows",2),("duplicate_rows_elided",1),("miss_rows",missing)):
                    inc(k,n)
                inc("native_requested_rows",missing);inc("native_selected_bytes",missing*160)
            else:
                inc("bypass_calls");inc("native_requested_rows",3);inc("native_selected_bytes",480)
        rows["retained_bytes"]=sum(t["nbytes"] for t in rows["live_tables"])
        lengths=a.chunks(p);n=len(lengths) if enabled else 0
        cone={"enabled":enabled,"closed":False,"owned_instances":100,"counts":{"model_calls":len(lengths)+3},
              "cone_counts":{"cone_calls":n} if enabled else {},"shape_counts":{},
              "prime_counts":{"eligible_prefills":n,"padded_appends":n,"wkv_pads":3*n} if enabled else {},
              "trace":[],"prefill_trace":[]}
        if not enabled:cone["counts"]["disabled_calls"]=len(lengths)+3
        start=0
        for chunk in lengths:
            if not enabled:
                cone["prefill_trace"].append({"cone":[],"shape":[],"prime":[]});start+=chunk;continue
            end=start+chunk
            layers=[];inrows=chunk
            for layer in range(20,40):
                tail=128+(40-layer)*127
                layers.append({"layer":layer,"input_rows":inrows,"query_rows":tail,"input_start":end-inrows,
                               "query_start":end-tail,"endpoint":end})
                inrows=tail
            ce={"mode":"cone","reason":None,"input_tokens":chunk,"start":start,"end":end,"chunk_tokens":chunk,"next_layer":40,"layers":layers}
            pe={"native_rows":chunk,"complete":True,"reason":None,"start":start,"end":end,"append_start":end-128,
                "append_count":1,"append_complete":True,"stages":[{"stage":j,"input_rows":128,"native_rows":chunk,"output_rows":128} for j in range(3)]}
            shapes=[{"kind":k,"native_rows":chunk} for k,count in (("index-pad",5),("gate-pad",20),("final-moe-256",1)) for _ in range(count)]
            cone["prefill_trace"].append({"cone":[ce],"shape":shapes,"prime":[pe]});start=end
        prefill=(p["prompt_tokens"]/16384)*(5 if enabled else 10)
        ttft=prefill+.1
        arrival=[ttft+j/40 for j in range(128)]
        total=arrival[-1]+.01
        activation={"expected_cone_chunks":n,"actual_cone_chunks":n,"actual_padded_appends":n,"actual_stage_pads":3*n,
                    "row_cache_unique_queries":4 if enabled else 0,"retained_cache_bytes":rows["retained_bytes"],
                    "prefill_trace_count":len(lengths),"prefill_trace_evictions":0,"complete_chunk_traces":True,"passed":True}
        row={"type":"run","profile":runtime|{"name":job["profile"]},"prompt_id":job["prompt"],"repeat":0,"seed":job["seed"],
             "warmup":job["warmup"],"input_tokens":p["prompt_tokens"],"input_tokens_sha256":p["input_tokens_sha256"],
             "generated_tokens":128,"token_ids":list(range(128)),"text":"synthetic equal response","finish_reason":"length",
             "fixed_length_valid":True,"first_eos_position":None,"prefill_seconds":prefill,"ttft_seconds":ttft,
             "total_seconds":total,"decode_tokens_per_second":127/(arrival[-1]-ttft),"arrival_seconds":arrival,
             "resources":{"process_footprint_peak_bytes":2000,"system_available_min_bytes":500,"swap_start_bytes":0,
                          "swap_peak_bytes":0,"swap_end_bytes":0,"disk_read_bytes_delta":100 if enabled else 200,"disk_write_bytes_delta":0,"mlx_peak_bytes":1900},
             "cache_state":"fresh KV; synthetic fixture", "events":[{"cycle_ms":1 if enabled else 2,"drafted":5,"accepted":3}],
             "mtp_stats":[{"cycles":1,"draft_emits":127,"init_emits":1,"bonus_emits":0,"verify_emits":0}],
             "qualification":{"job_index":i,"pair_index":i//2,"worker_thread":q["worker_threads"][(i//2)%2],
                              "prefill_route":"VLMModelAdapter" if (i//2)%2==0 else "LanguageModel._omlx_prefill",
                              "requested_output_tokens":128,"activation":activation},
             "persistent_adapter":{"before":{"rows":before},"after":{"rows":copy.deepcopy(rows),"cone":cone}}}
        records.append(row)
        if i%2:
            x,y=records[-2:];x,y=(x,y) if x["profile"]["name"]=="native64" else (y,x)
            q["pairs"].append({"checks":a.check_pair(x,y),"passed":True,"compared_output_tokens":128,"compared_mtp_cycles":1,
                               "prompt_id":p["id"],"warmup":job["warmup"],"pair_index":i//2})
    final=copy.deepcopy(rows);final.update(status="closed",enabled=False,retained_bytes=0,retired_tables=final["live_tables"],live_tables=[])
    for t in final["retired_tables"]:t.update(closed=True,nbytes=0)
    q["row_cache_after_model_close"]=final
    return records,plan,profile,q


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n')


def archive_fixture(folder, data):
    folder = folder.resolve()
    records,plan,profile,q=data
    live=folder/'live'; out=folder/'results/run.jsonl';out.parent.mkdir(parents=True)
    plan_path=live/'performance.json';manifest_path=live/'execution.json';profile_path=live/'qualified-profile.json'
    checkpoint_path=live/'results/checkpoint-hashes.json';fixture_path=live/'fixture.json';source_path=live/'sources.json'
    write_json(source_path,{"source":"synthetic"})
    fixtures={"baseline_runtime":profile['runtime'],"source_manifest_sha256":a.sha(source_path)}
    write_json(fixture_path,fixtures)
    plan.update(fixture_manifest=str(fixture_path),fixture_manifest_sha256=a.sha(fixture_path),source_manifest_sha256=a.sha(source_path))
    write_json(plan_path,plan);write_json(profile_path,profile)
    checkpoint={"repo":"mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp","revision":"78456115d301eecf9fadcdcc81c0cf6f8b2c8eab","status":"complete_hash_verified","missing":[],"directory":"/fixture/model","files":{"weights":{"verified":True,"fingerprint":[1,2,3]}}}
    write_json(checkpoint_path,checkpoint)
    files=[plan_path,profile_path,checkpoint_path,fixture_path,source_path]
    manifest={"sources_sha256":{str(p):a.sha(p) for p in files},"baseline":{"profile_sha256":a.sha(profile_path),
              "checkpoint_manifest_sha256":a.sha(checkpoint_path),"packages":profile['packages'],
              "inference_files_verified":len(profile['inference_files_sha256']),"checkpoint_fingerprints_verified":1}}
    write_json(manifest_path,manifest)
    archive=out.with_suffix('.sources');archive.mkdir()
    index={}
    for i,p in enumerate(files+[manifest_path]):
        name=f'{i:03d}-{p.name}';(archive/name).write_bytes(p.read_bytes())
        index[str(p)]={"snapshot":name,"sha256":a.sha(p)}
    write_json(archive/'manifest.json',index)
    q.update(plan_sha256=a.sha(plan_path),manifest_sha256=a.sha(manifest_path),completed_request_files_sha256={})
    completed=out.with_suffix('.completed');completed.mkdir()
    for i,row in enumerate(records[2:]):
        backup=copy.deepcopy({k:v for k,v in row.items() if k not in ('persistent_adapter','warmup')})
        backup['qualification'].pop('activation')
        p=completed/f'job{i:02d}.json';write_json(p,backup);q['completed_request_files_sha256'][str(p)]=a.sha(p)
    out.write_text(''.join(json.dumps(r)+'\n' for r in records));q['raw_output_sha256']=a.sha(out)
    write_json(out.with_suffix('.qualification.json'),q)
    return out,plan_path,manifest_path


class AnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base=fixture()
    def setUp(self):self.data=copy.deepcopy(self.base)
    def validate(self):return a.validate_records(*self.data)
    def reject(self):
        with self.assertRaises((ValueError,KeyError,TypeError)):self.validate()
    def test_complete_expected_aggregate_and_separate_warmup(self):
        pairs=self.validate();report=a.build_report(pairs)
        self.assertEqual(report['measured']['pairs'],12);self.assertEqual(report['warmup_separate']['pairs'],1)
        self.assertAlmostEqual(report['measured']['prefill_seconds']['paired_speed_ratio_geomean'],2)
        self.assertAlmostEqual(report['measured']['prefill_seconds']['latency_reduction_percent'],50)
        self.assertEqual(report['all_measured_orders'],{'AB':6,'BA':6})
        self.assertEqual(report['measured']['output_tokens_compared'],1536)
        self.assertEqual(report['by_input_tokens']['131072']['pairs'],1)
    def test_partial_rejected(self):self.data[0].pop();self.reject()
    def test_error_record_rejected(self):self.data[0].append({'type':'error'});self.reject()
    def test_order_rejected(self):self.data[0][4],self.data[0][6]=self.data[0][6],self.data[0][4];self.reject()
    def test_same_name_changed_seed_rejected(self):self.data[0][4]['seed']+=1;self.reject()
    def test_profile_changed_rejected(self):self.data[0][4]['profile']['engram_prefetch_bytes']=16*2**20;self.reject()
    def test_input_hash_changed_rejected(self):self.data[0][4]['input_tokens_sha256']='bad';self.reject()
    def test_early_eos_rejected(self):self.data[0][4]['first_eos_position']=12;self.reject()
    def test_output_length_rejected(self):self.data[0][4]['token_ids'].pop();self.reject()
    def test_token_divergence_rejected(self):self.data[0][4]['token_ids'][5]=777;self.reject()
    def test_text_divergence_rejected(self):self.data[0][4]['text']='different';self.reject()
    def test_mtp_functional_divergence_rejected(self):self.data[0][4]['events'][0]['accepted']=2;self.reject()
    def test_mtp_emission_accounting_rejected(self):self.data[0][4]["mtp_stats"][0]["draft_emits"]-=1;self.reject()
    def test_nonfinite_json_rejected(self):
        with self.assertRaises(ValueError):a.loads('{"x":NaN}')
    def test_mtp_timing_difference_allowed(self):self.data[0][4]['events'][0]['cycle_ms']=300;self.validate()
    def test_reported_boolean_cannot_hide_missing_stage(self):
        row=next(r for r in self.data[0][4:] if r['profile']['name']=='fastpath64')
        row['persistent_adapter']['after']['cone']['prefill_trace'][0]['prime'][0]['stages'].pop();self.reject()
    def test_invalid_absolute_endpoint_rejected(self):
        row=next(r for r in self.data[0][4:] if r['profile']['name']=='fastpath64')
        row['persistent_adapter']['after']['cone']['prefill_trace'][0]['cone'][0]['layers'][7]['endpoint']+=1;self.reject()
    def test_trace_eviction_rejected(self):
        self.data[0][5]['persistent_adapter']['after']['cone']['counts']['prefill_trace_evictions']=1;self.reject()
    def test_cache_history_reset_rejected(self):
        self.data[0][5]['persistent_adapter']['before']['rows']['live_tables'][0]['unique_queries']=0;self.reject()
    def test_cache_byte_limit_rejected(self):
        self.data[0][5]['persistent_adapter']['after']['rows']['live_tables'][0]['nbytes']=a.TABLE_BUDGET+1;self.reject()
    def test_native_io_inconsistency_rejected(self):
        self.data[0][5]['persistent_adapter']['after']['rows']['live_tables'][0]['counts']['native_selected_bytes']+=1;self.reject()
    def test_final_cleanup_rejected(self):self.data[3]['row_cache_after_model_close']['retained_bytes']=1;self.reject()
    def test_status_incomplete_rejected(self):self.data[3]['status']='running';self.reject()
    def test_false_pair_summary_rejected(self):self.data[3]['pairs'][2]['checks']['same_seed']=False;self.reject()
    def test_missing_parameter_source_check_rejected(self):self.data[3]['source_hashes_reverified']=False;self.reject()
    def test_worker_thread_routing_rejected(self):self.data[0][4]['qualification']['worker_thread']=999;self.reject()
    def test_nonfinite_timing_rejected(self):self.data[0][4]['prefill_seconds']=float('nan');self.reject()
    def test_boundary_must_not_count_as_performance(self):self.data[1]['design']['suite']='boundary';self.reject()
    def test_extra_unqualified_python_metadata_allowed(self):
        self.data[0][0]["experiment_files_sha256"]["reporting.py"]="d"*64
        self.validate()
    def test_qualified_bench_pin_mismatch_rejected(self):
        self.data[0][0]["experiment_files_sha256"]["bench.py"]="d"*64
        self.reject()
    def test_malformed_extra_source_metadata_rejected(self):
        self.data[0][0]["experiment_files_sha256"]["../reporting.py"]="d"*64
        self.reject()
    def test_raw_runtime_pin_rejected(self):self.data[0][0]['runtime_files_sha256']['test.py']='bad';self.reject()
    def test_archived_roundtrip_ignores_changed_live_profile(self):
        with tempfile.TemporaryDirectory() as name:
            paths=archive_fixture(Path(name),self.data)
            (Path(name)/'live/qualified-profile.json').write_text('changed after run')
            records,plan,profile,q,pins=a.archived_evidence(*paths)
            a.validate_records(records,plan,profile,q)
            self.assertTrue(pins['source_files_verified']>0)
            self.assertNotIn('mlx.core',sys.modules)
    def test_archived_bytes_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            paths=archive_fixture(Path(name),self.data)
            (paths[0].with_suffix('.sources')/'000-performance.json').write_text('{}')
            with self.assertRaises(ValueError):a.archived_evidence(*paths)
    def test_raw_hash_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            paths=archive_fixture(Path(name),self.data);paths[0].write_text(paths[0].read_text()+'\n')
            with self.assertRaises(ValueError):a.archived_evidence(*paths)
    def test_completed_backup_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            paths=archive_fixture(Path(name),self.data)
            (paths[0].with_suffix('.completed')/'job02.json').write_text('{}')
            with self.assertRaises(ValueError):a.archived_evidence(*paths)


if __name__=='__main__':unittest.main(verbosity=2)
