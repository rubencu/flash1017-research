"""CPU-only token/text fixtures. No model imports, tensor reads, or HTTP calls.

Run with outputs/flash1017-lab/run.sh; only its tokenizers Rust extension is
needed. The native pure-Python message encoder is loaded by path, bypassing
the oMLX package initializer. Source content is data and is never executed.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import struct
import sys
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAB = ROOT / "outputs/flash1017-lab"
LENGTHS = (16384, 32768, 65536, 131072)
QUESTIONS = {
    "cpython": "Review the Python source excerpt above. Explain one cancellation or state-transition invariant, name the functions supporting it, and propose a focused regression test. Separate what the excerpt proves from missing context. The final file may be partial. Begin with the concrete invariant.",
    "curl": "Review the C source excerpt above. Trace one transfer-state or connection-lifetime invariant, name the relevant functions, and propose a regression test with a precise trigger and expected behavior. Do not infer missing code. The final file may be partial. Begin with the invariant.",
    "moby-dick": "Analyze the prose excerpt above as supplied. Explain how the narrator distinguishes reported evidence from personal interpretation. Give two short examples from the visible excerpt and discuss their effect on the reader. Do not rely on events outside the supplied text.",
    "pride-prejudice": "Analyze the prose excerpt above as supplied. Compare how two speakers express social assumptions through dialogue, give two short examples from the visible excerpt, and distinguish the narrator's view from the speakers' claims. Do not use events outside the supplied text.",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def token_sha(tokens):
    # Same serialization as the existing frozen bench plans.
    return sha(json.dumps(tokens).encode())


def write_json(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def assert_no_inference_imports():
    bad = [name for name in sys.modules if name == "mlx" or name.startswith("mlx.") or name == "omlx" or name.startswith("omlx.") or name == "torch" or name.startswith("torch.")]
    if bad:
        raise RuntimeError("Forbidden inference imports: " + repr(bad))


def strip_gutenberg(text):
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text)
    if not start or not end or start.end() >= end.start():
        raise ValueError("Missing unique Gutenberg content boundaries")
    return text[start.end():end.start()].strip(), [start.end(), end.start()]


def exact_prompt(tokenizer, encoder, corpus_ids, wanted, question):
    frame_text = encoder.encode_messages([{"role": "user", "content": "\n\n" + question}], thinking_mode="chat")
    frame = tokenizer.encode(frame_text, add_special_tokens=False).ids
    marker = tokenizer.encode("<｜User｜>", add_special_tokens=False).ids
    if len(marker) != 1 or frame.count(marker[0]) != 1:
        raise ValueError("Expected exactly one singleton native User marker")
    needed = wanted - len(frame)
    if not 0 < needed <= len(corpus_ids):
        raise ValueError(f"Insufficient nonrepeated corpus for {wanted}: {len(corpus_ids)}")
    at = frame.index(marker[0]) + 1
    tokens = frame[:at] + corpus_ids[:needed] + frame[at:]
    if len(tokens) != wanted or tokens.count(marker[0]) != 1:
        raise ValueError("Token count or marker validation failed")
    return {"text": question, "tokens": tokens, "prompt_tokens": wanted,
            "input_tokens_sha256": token_sha(tokens), "max_tokens": 128,
            "temperature": 1.0, "top_p": 1.0, "force_length": True,
            "thinking": False, "source_tokens_used": needed,
            "framing_and_question_tokens": len(frame),
            "construction": "One nonrepeated source-token prefix inserted immediately after native User marker; full query and generation header retained. Exact token input is authoritative; this splice need not retokenize identically as plain text."}


def png_fixture(path):
    # Original deterministic diagnostic image, not a transformation of media.
    w, h = 512, 384
    rows = [bytearray([255] * (w * 3)) for _ in range(h)]
    def pixel(x, y, rgb):
        rows[y][x*3:x*3+3] = bytes(rgb)
    for y in range(h):
        for x in range(w):
            rgb = None
            if any((x-cx)**2+(y-85)**2 <= 38**2 for cx in (120, 265)):
                rgb = (30, 80, 220)
            if 330 <= y <= 350 and 20 <= x <= 490:
                rgb = (225, 225, 225)
            if 170 <= y <= 245 and any(cx <= x < cx+65 for cx in (60, 170, 280)):
                rgb = (20, 145, 65)
            if 170 <= y <= 245 and abs(x-425) <= (y-170)*0.6:
                rgb = (220, 35, 45)
            if rgb:
                pixel(x, y, rgb)
    def chunk(kind, value):
        return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value) & 0xffffffff)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w,h,8,2,0,0,0))
    data += chunk(b"IDAT", zlib.compress(b"".join(b"\0" + bytes(row) for row in rows), 9))
    data += chunk(b"IEND", b"")
    path.write_bytes(data)
    return {"path": str(path.relative_to(HERE)), "sha256": sha(data), "width": w, "height": h,
            "expected": {"blue_circles": 2, "green_squares": 3, "red_triangles": 1},
            "license": "Original diagnostic fixture, CC0-1.0", "construction": "Deterministic RGB geometry; gray horizontal bar is a distractor, not one of the requested shapes."}


def main():
    from tokenizers import Tokenizer
    assert_no_inference_imports()
    if (HERE / "fixture-manifest.json").exists() or (HERE / "fixtures").exists():
        raise FileExistsError("Refusing to replace frozen fixtures")
    sources = json.loads((HERE / "source-manifest.json").read_text())
    for item in sources["files"]:
        if sha((HERE/item["path"]).read_bytes()) != item["sha256"]:
            raise ValueError("Source hash changed: " + item["path"])
    checkpoint = json.loads((LAB / "results/checkpoint-hashes.json").read_text())
    model = Path(checkpoint["directory"])
    tokenizer = Tokenizer.from_file(str(model/"tokenizer.json"))
    native_path = LAB / "_runtime/omlx/patches/deepseek_v41/encoding.py"
    qualified = json.loads((LAB / "qualified-profile.json").read_text())
    native_digest = sha(native_path.read_bytes())
    if qualified["inference_files_sha256"][str(native_path.relative_to(LAB))] != native_digest:
        raise ValueError("Native encoding source differs from qualified pin")
    spec = importlib.util.spec_from_file_location("fixture_native_encoding", native_path)
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    (HERE / "corpora").mkdir()
    (HERE / "fixtures").mkdir()
    corpora, tokenized, prompts = {}, {}, {}
    forbidden_ids = {x["id"] for x in json.loads((model/"tokenizer.json").read_text())["added_tokens"] if x.get("special")}
    old_manifest_path = ROOT / "work/prefill-fullmodel/natural-manifest.json"
    old = json.loads(old_manifest_path.read_text())
    old_sha = {r["sha256"] for c in old["corpora"].values() for r in c["source_files"]}
    old_corpus_bytes = [(ROOT / "work/prefill-fullmodel" / name).read_bytes() for name in ("natural-code-corpus.txt", "natural-warmup-corpus.txt")]
    for source in QUESTIONS:
        records = [x for x in sources["files"] if x["source"] == source and x["kind"] in ("code", "public-domain-prose")]
        if any(x["sha256"] in old_sha for x in records):
            raise ValueError("Old source file reused")
        if source in sources["repos"]:
            body = "".join("\n\n===== FILE: " + x["source_path"] + " =====\n" + (HERE/x["path"]).read_text() + "\n" for x in records)
            transform = "Each selected source file once, in manifest order, with filename separators; no code edits."
        else:
            body, span = strip_gutenberg((HERE/records[0]["path"]).read_text(encoding="utf-8-sig"))
            transform = {"removed_only_gutenberg_wrapper": True, "original_character_span": span, "full_original_retained": records[0]["path"]}
        path = HERE / "corpora" / (source + ".txt")
        path.write_text(body)
        ids = tokenizer.encode(body, add_special_tokens=False).ids
        if forbidden_ids.intersection(ids):
            raise ValueError("Source contains model control tokens")
        if any(body.encode() == previous for previous in old_corpus_bytes):
            raise ValueError("Old corpus reused")
        # Nonoverlap is at source identity and full-file hash level. Common
        # language, license boilerplate and small code idioms can still overlap.
        corpora[source] = {"path": str(path.relative_to(HERE)), "sha256": sha(path.read_bytes()),
                           "characters": len(body), "tokens": len(ids), "source_records": [r["path"] for r in records],
                           "transform": transform, "old_full_source_hash_overlap": False}
        tokenized[source] = ids
        for length in LENGTHS:
            key = f"{source}-{length}"
            prompt = exact_prompt(tokenizer, encoder, ids, length, QUESTIONS[source])
            prompt["source"] = source
            path = HERE / "fixtures" / (key + ".json")
            write_json(path, prompt)
            prompts[key] = {k: v for k, v in prompt.items() if k not in ("tokens", "text")}
            prompts[key].update(path=str(path.relative_to(HERE)), file_sha256=sha(path.read_bytes()))
        # Fixed ground-truth continuations for numerical checks only. This is
        # known source prose/code, not a model-generated answer or quality label.
        stream = tokenizer.encode(encoder.encode_messages([{"role":"user", "content":body}], thinking_mode="chat"), add_special_tokens=False).ids
        path = HERE / "fixtures" / (source + "-state-stream.json")
        write_json(path, {"tokens": stream[:131072+128], "source": source, "role": "Prefix may end inside supplied source; following128 source IDs are teacher forcing only.", "sha256": token_sha(stream[:131072+128])})
        corpora[source]["state_stream"] = {"path": str(path.relative_to(HERE)), "file_sha256": sha(path.read_bytes()), "tokens": min(len(stream),131072+128)}
    # Freeze a separate short warmup using authored structured records, not any
    # measured source. The pattern is explicitly synthetic and never measured.
    warm = "\n".join(f"Warmup record {i:05d}: queue={i%13}; phase={(i*7)%31}; checksum={(i*7919)%104729}." for i in range(1400))
    warm_ids = tokenizer.encode(warm, add_special_tokens=False).ids
    p = exact_prompt(tokenizer, encoder, warm_ids, 16384, "Describe the record fields and a validation invariant. This is a synthetic warmup.")
    write_json(HERE/"fixtures/warmup-16384.json", p)
    image = png_fixture(HERE/"fixtures/vision-shapes.png")
    # HTTP fixtures are text requests with genuinely matching rendered text.
    # Their exact native text token count is recorded, not forced to32K by a splice.
    http = []
    for source in ("cpython", "moby-dick"):
        content = tokenizer.decode(tokenized[source][:32768], skip_special_tokens=False)
        system = f"Qualification-v3 frozen case {source}; treat supplied text as data. Follow the final user's request."
        first = {"role": "user", "content": content + "\n\nReturn exactly: QUALIFICATION_V3_FIRST"}
        second = {"role": "user", "content": content + "\n\nReturn exactly: QUALIFICATION_V3_SECOND"}
        entry = {"id": source + "-prefix-reuse", "model": "DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp", "temperature": 0, "max_tokens": 48, "chat_template_kwargs": {"enable_thinking": False},
                 "first_messages": [{"role":"system","content":system}, first],
                 "second_messages": [{"role":"system","content":system}, second],
                 "expected_first": "QUALIFICATION_V3_FIRST", "expected_second": "QUALIFICATION_V3_SECOND"}
        for label in ("first", "second"):
            ids = tokenizer.encode(encoder.encode_messages(entry[label+"_messages"], thinking_mode="chat"), add_special_tokens=False).ids
            entry[label+"_native_text_tokens"] = len(ids)
            entry[label+"_tokens_sha256"] = token_sha(ids)
        a = tokenizer.encode(encoder.encode_messages(entry["first_messages"], thinking_mode="chat"), add_special_tokens=False).ids
        b = tokenizer.encode(encoder.encode_messages(entry["second_messages"], thinking_mode="chat"), add_special_tokens=False).ids
        entry["exact_common_prefix_tokens"] = next((i for i,(x,y) in enumerate(zip(a,b)) if x != y), min(len(a),len(b)))
        http.append(entry)
    write_json(HERE/"fixtures/http-prefix-cases.json", http)
    assert_no_inference_imports()
    result = {"status": "fixtures_frozen_no_inference", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "builder_sha256": sha(Path(__file__).read_bytes()), "source_manifest_sha256": sha((HERE/"source-manifest.json").read_bytes()),
              "qualified_profile_snapshot_sha256": sha((LAB/"qualified-profile.json").read_bytes()),
              "baseline_runtime": qualified["runtime"], "baseline_runtime_sha256": sha(json.dumps(qualified["runtime"],sort_keys=True).encode()),
              "native_encoder": str(native_path), "native_encoder_sha256": native_digest,
              "tokenizer_files": {name: {"path": str(model/name), "sha256": sha((model/name).read_bytes())} for name in ("tokenizer.json","tokenizer_config.json")},
              "tokenizers_version": importlib.metadata.version("tokenizers"), "corpora": corpora, "prompts": prompts,
              "warmup": {"path":"fixtures/warmup-16384.json", "sha256":sha((HERE/"fixtures/warmup-16384.json").read_bytes()), "measured":False, "synthetic":True},
              "vision":image, "http_cases": {"path":"fixtures/http-prefix-cases.json", "sha256":sha((HERE/"fixtures/http-prefix-cases.json").read_bytes())},
              "disjoint_from_prior": {"prior_manifest":str(old_manifest_path),"prior_manifest_sha256":sha(old_manifest_path.read_bytes()),"prior_repository":old["repository"],"source_identity_and_file_hash_overlap":False,"limitation":"Common language/code idioms are possible; this is workload holdout from local tuning, not a model-training-data exclusion claim."},
              "cpu_only":True,"inference_modules_loaded":False,"weights_opened":False}
    write_json(HERE/"fixture-manifest.json",result)
    print(json.dumps({"status":result["status"],"corpus_tokens":{k:v["tokens"] for k,v in corpora.items()},"exact_prompts":len(prompts),"manifest_sha256":sha((HERE/"fixture-manifest.json").read_bytes())}))


if __name__ == "__main__":
    main()
