"""Candidate text, reasoning, tool formatting, prefix reuse and queued clients."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import threading
import time
from common import Client,FIX,content,delta,envelope,sha,write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    client=Client(args.profile,"candidate")
    report=envelope(client,"sessions")
    fixture=FIX/"fixtures/http-prefix-cases.json"
    report["prefix_fixture_sha256"]=sha(fixture)
    write(args.output,report)
    def save(name,row,passed):
        row.update(test=name,passed=bool(passed))
        report["tests"].append(row)
        write(args.output,report)
        print(json.dumps({"test":name,"passed":bool(passed)}),flush=True)
    try:
        row=client.chat([{"role":"user","content":"Reply exactly: SERVING_V3_READY"}],max_tokens=32)
        save("short-text",row,content(row["response"])=="SERVING_V3_READY")
        row=client.chat([{"role":"user","content":"A shelf has 17 boxes, each containing 19 books. How many books are there? Give the final number."}],
            max_tokens=512,reasoning_effort="low",chat_template_kwargs={"enable_thinking":True})
        message=row["response"]["choices"][0]["message"]
        save("reasoning",row,bool(message.get("reasoning_content")) and re.search(r"\b323\b",content(row["response"])) is not None)
        tools=[{"type":"function","function":{"name":"lookup_inventory","description":"Read inventory count for a product SKU.",
            "parameters":{"type":"object","properties":{"sku":{"type":"string"}},"required":["sku"],"additionalProperties":False}}}]
        messages=[{"role":"user","content":"Use lookup_inventory to check SKU DS1017-V3, then tell me its stock count."}]
        row=client.chat(messages,tools=tools,tool_choice={"type":"function","function":{"name":"lookup_inventory"}})
        assistant=row["response"]["choices"][0]["message"]
        calls=assistant.get("tool_calls") or []
        valid=False
        try:
            valid=(len(calls)==1 and calls[0]["function"]["name"]=="lookup_inventory" and
                   json.loads(calls[0]["function"]["arguments"])=={"sku":"DS1017-V3"})
        except (ValueError,IndexError,KeyError):pass
        save("tool-formatting",row,valid)
        if valid:
            row=client.chat(messages+[assistant,{"role":"tool","tool_call_id":calls[0]["id"],"content":'{"sku":"DS1017-V3","stock":47}'}],tools=tools)
            save("tool-continuation",row,re.search(r"\b47\b",content(row["response"])) is not None)
        for case in json.loads(fixture.read_text()):
            first=client.chat(case["first_messages"],max_tokens=case["max_tokens"])
            second=client.chat(case["second_messages"],max_tokens=case["max_tokens"])
            cache=lambda r:r["response"].get("usage",{}).get("prompt_tokens_details",{}).get("cached_tokens",0)
            first_delta,second_delta=delta(first),delta(second)
            prime_delta=delta(first,"prime_counts")
            second_prime=delta(second,"prime_counts")
            residual=second["response"]["usage"]["prompt_tokens"]-cache(second)
            valid=(content(first["response"])==case["expected_first"] and content(second["response"])==case["expected_second"]
                and cache(first)==0 and cache(second)>16384 and first_delta.get("cone_calls",0)>0
                and prime_delta.get("padded_appends",0)==first_delta.get("cone_calls",0)
                and prime_delta.get("wkv_pads",0)==3*first_delta.get("cone_calls",0)
                and second_delta.get("fallback:short-chunk",0)>0 and second_delta.get("cone_calls",0)==0
                and second_prime.get("fallback:short-chunk",0)>0
                and second_prime.get("padded_appends",0)==second_prime.get("wkv_pads",0)==0
                and 2 <= residual < 2668)
            save(case["id"],{"first":first,"second":second,"first_cached_tokens":cache(first),"second_cached_tokens":cache(second),
                "first_cone_delta":first_delta,"second_cone_delta":second_delta,"second_uncached_tokens":residual,
                "prime_delta":prime_delta,"second_prime_delta":second_prime},valid)
        before=client.telemetry()
        barrier=threading.Barrier(3)
        def queued(marker):
            barrier.wait(timeout=15)
            return client.chat([{"role":"user","content":"Reply exactly: "+marker}],max_tokens=32)
        observed=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(queued,marker) for marker in ("QUEUED_V3_ALPHA","QUEUED_V3_BETA")]
            barrier.wait(timeout=15)
            started=time.monotonic()
            while not all(f.done() for f in futures):
                observed.append({"seconds":time.monotonic()-started,**client.status()})
                time.sleep(.01)
            rows=[future.result() for future in futures]
        after=client.telemetry()
        failures=lambda t:t["models"][0]["call_counts"].get("call_failures",0)
        queued_observed=any(s["active_requests"]>=2 and s["waiting_requests"]>=1 for s in observed)
        save("two-queued-clients",{"responses":rows,"before":before,"after":after,"status_observations":observed,
             "actual_queue_observed":queued_observed},
            [content(r["response"]) for r in rows]==["QUEUED_V3_ALPHA","QUEUED_V3_BETA"]
            and queued_observed and failures(after)==failures(before))
        if not all(x["passed"] for x in report["tests"]):raise RuntimeError("A serving session check failed")
        if len(report["tests"]) != 7:raise RuntimeError("A required serving check was skipped")
        client.verify()
        report["status"]="complete_validated"
    except BaseException as exc:
        report.update(status="failed",exception=repr(exc))
        raise
    finally:
        report["completed_at_unix"]=time.time()
        write(args.output,report)


if __name__=="__main__":main()
