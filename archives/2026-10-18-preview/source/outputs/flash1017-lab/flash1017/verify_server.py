"""Exercise the actual local HTTP path: text, tools, vision and prefix reuse."""
import argparse
import base64
import json
from pathlib import Path
import re
import time
import urllib.request


def create_image(path):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    for x, y in ((100, 100), (430, 120), (170, 430)):
        draw.ellipse((x, y, x + 140, y + 140), fill=(230, 20, 20))
    for x, y in ((710, 100), (670, 420)):
        draw.rectangle((x, y, x + 150, y + 150), fill=(20, 60, 230))
    draw.polygon([(470, 390), (380, 570), (560, 570)], fill=(20, 170, 50))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def text(response):
    return response["choices"][0]["message"].get("content") or ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    hashes = json.loads((root / "results/checkpoint-hashes.json").read_text())
    model_id = Path(hashes["directory"]).name

    def chat(messages, **extra):
        payload = {"model": model_id, "messages": messages, "temperature": 0,
                   "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}, **extra}
        request = urllib.request.Request(args.url + "/v1/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        return {"response": result, "seconds": time.perf_counter() - started,
                "request": payload}

    results = []

    def save(name, result, passed):
        row = {"test": name, "passed": bool(passed), **result}
        # Store no redundant base64 blob; retain its deterministic image path.
        for message in row.get("request", {}).get("messages", []):
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item.get("type") == "image_url":
                        item["image_url"]["url"] = "fixtures/vision-grid.png"
        results.append(row)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"test": name, "passed": bool(passed), "seconds": row.get("seconds")}), flush=True)

    identity = chat([{"role": "user", "content": "Reply exactly: FLASH1017_OK"}], max_tokens=32)
    save("text_identity", identity, text(identity["response"]).strip() == "FLASH1017_OK")
    thinking = chat([{"role": "user", "content": "A shelf has 17 boxes, each containing 19 books. How many books are there? Give the final number."}],
                    max_tokens=512, reasoning_effort="low",
                    chat_template_kwargs={"enable_thinking": True})
    thought_message = thinking["response"]["choices"][0]["message"]
    save("reasoning_channel", thinking,
         bool(thought_message.get("reasoning_content"))
         and re.search(r"\b323\b", text(thinking["response"])) is not None)
    tools = [{"type": "function", "function": {
        "name": "lookup_inventory", "description": "Read the inventory count for a product SKU.",
        "parameters": {"type": "object", "properties": {"sku": {"type": "string"}},
                       "required": ["sku"], "additionalProperties": False}}}]
    messages = [{"role": "user", "content": "Use lookup_inventory to check SKU DS1017-A, then tell me its stock count."}]
    call = chat(messages, tools=tools,
                tool_choice={"type": "function", "function": {"name": "lookup_inventory"}})
    assistant = call["response"]["choices"][0]["message"]
    calls = assistant.get("tool_calls") or []
    valid = len(calls) == 1 and calls[0]["function"]["name"] == "lookup_inventory"
    try:
        valid = valid and json.loads(calls[0]["function"]["arguments"]) == {"sku": "DS1017-A"}
    except (ValueError, IndexError, KeyError):
        valid = False
    save("tool_call", call, valid)
    if valid:
        followup = chat(messages + [assistant, {"role": "tool", "tool_call_id": calls[0]["id"],
                                                "content": '{"sku":"DS1017-A","stock":31}'}], tools=tools)
        save("tool_continuation", followup, re.search(r"\b31\b", text(followup["response"])) is not None)

    image = root / "fixtures/vision-grid.png"
    create_image(image)
    image_data = base64.b64encode(image.read_bytes()).decode()
    vision = chat([{"role": "user", "content": [
        {"type": "text", "text": 'Count the shapes. Return only JSON with integer values for the keys "red_circles", "blue_squares", and "green_triangles".'},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_data}},
    ]}])
    try:
        match = re.search(r"\{.*?\}", text(vision["response"]), re.DOTALL)
        parsed = json.loads(match.group()) if match else None
    except ValueError:
        parsed = None
    save("vision_shapes", vision, parsed == {"red_circles": 3, "blue_squares": 2, "green_triangles": 1})

    context = "\n".join(f"Warehouse record {i}: bins are numbered, inventory is checked each evening." for i in range(240))
    prompt = [{"role": "user", "content": context + "\nReply exactly: CACHE1017_OK"}]
    first = chat(prompt, max_tokens=32)
    second = chat(prompt, max_tokens=32)
    cached = second["response"].get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    save("prefix_reuse", {"first": first, "second": second, "cached_tokens": cached},
         text(first["response"]).strip() == "CACHE1017_OK"
         and text(second["response"]) == text(first["response"]) and cached > 0)

    # A local user must be able to stop a response and immediately ask another
    # question. Close a real stream after its first visible text, then recover.
    payload = {"model": model_id, "messages": [{"role": "user", "content": "Write a very detailed implementation guide for a versioned document database."}],
               "temperature": 0, "max_tokens": 4096, "stream": True,
               "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(args.url + "/v1/chat/completions",
              data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    saw_text = False
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            item = json.loads(line[6:])
            if any(c.get("delta", {}).get("content") for c in item.get("choices", [])):
                saw_text = True
                break
    deadline = time.monotonic() + 30
    idle = False
    while time.monotonic() < deadline:
        with urllib.request.urlopen(args.url + "/api/status", timeout=5) as response:
            status = json.load(response)
        if status.get("active_requests") == 0 and status.get("waiting_requests", 0) == 0:
            idle = True
            break
        time.sleep(.25)
    recovered = chat([{"role": "user", "content": "Reply exactly: CANCEL_RECOVERY_OK"}], max_tokens=32)
    save("cancel_and_recover", {"saw_text_before_cancel": saw_text, "became_idle": idle, "recovery": recovered},
         saw_text and idle and text(recovered["response"]).strip() == "CANCEL_RECOVERY_OK")


if __name__ == "__main__":
    main()
