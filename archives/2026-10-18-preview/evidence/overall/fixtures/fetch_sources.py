"""Download small public text fixtures; never import an inference package."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

DEST = Path(__file__).resolve().parent
REPOS = {
    "cpython": {"repo": "python/cpython", "revision": "ebf955df7a89ed0c7968f79faec1de49f61ed7cb", "tag": "v3.14.0", "license": "PSF-2.0 and retained historical licenses", "files": [
        "LICENSE", "Lib/asyncio/base_events.py", "Lib/asyncio/events.py", "Lib/asyncio/tasks.py",
        "Lib/asyncio/futures.py", "Lib/asyncio/streams.py", "Lib/asyncio/locks.py",
        "Lib/asyncio/queues.py", "Lib/asyncio/taskgroups.py", "Lib/inspect.py", "Lib/ast.py",
        "Lib/dataclasses.py", "Lib/typing.py", "Lib/pickle.py", "Lib/dis.py"]},
    "curl": {"repo": "curl/curl", "revision": "11b991232fbcaa88e2b1faecac224416b0001e35", "tag": "curl-8_16_0", "license": "curl license (COPYING and per-file notices retained)", "files": [
        "COPYING", "lib/multi.c", "lib/easy.c", "lib/transfer.c", "lib/url.c", "lib/http.c",
        "lib/sendf.c", "lib/connect.c", "lib/hostip.c", "lib/urlapi.c", "lib/cfilters.c"]},
}
BOOKS = {
    "moby-dick": {"id": 2701, "title": "Moby Dick; Or, The Whale", "author": "Herman Melville", "url": "https://www.gutenberg.org/ebooks/2701.txt.utf-8"},
    "pride-prejudice": {"id": 1342, "title": "Pride and Prejudice", "author": "Jane Austen", "url": "https://www.gutenberg.org/ebooks/1342.txt.utf-8"},
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(job):
    source, relative, url, kind = job
    request = urllib.request.Request(url, headers={"User-Agent": "Flash1017-local-validation/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("Unexpectedly large text source")
        data.decode("utf-8-sig")
        headers = {k: response.headers.get(k) for k in ("ETag", "Last-Modified", "Content-Type")}
        final_url = response.url
    path = DEST / "sources" / source / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(data)
    return {"source": source, "path": str(path.relative_to(DEST)), "source_path": relative,
            "url": url, "final_url": final_url, "kind": kind, "bytes": len(data),
            "sha256": sha(data), "headers": headers}


def main():
    manifest = DEST / "source-manifest.json"
    if manifest.exists() or (DEST / "sources").exists():
        raise FileExistsError("Refusing to overwrite source fixtures")
    jobs = []
    for name, repo in REPOS.items():
        for path in repo["files"]:
            url = f"https://raw.githubusercontent.com/{repo['repo']}/{repo['revision']}/{path}"
            jobs.append((name, path, url, "license" if path in ("LICENSE", "COPYING") else "code"))
    for name, book in BOOKS.items():
        jobs.append((name, "complete.txt", book["url"], "public-domain-prose"))
        jobs.append((name, "catalog.html", f"https://www.gutenberg.org/ebooks/{book['id']}", "catalog-and-rights"))
    with ThreadPoolExecutor(max_workers=4) as executor:
        files = list(executor.map(fetch, jobs))
    result = {"fetched_at_utc": datetime.now(timezone.utc).isoformat(), "repos": REPOS,
              "books": BOOKS, "files": files, "fetcher_sha256": sha(Path(__file__).read_bytes()),
              "rights": "Code files retain full notices plus upstream license. Both Gutenberg catalog pages explicitly identify public-domain status in the USA; full ebook headers and license tails are retained. No inferred worldwide public-domain claim.",
              "purpose": "Held-out runtime-validation inputs from repositories/books absent from the earlier oMLX natural-source pilot. This does not imply absence from model training data."}
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest), "files": len(files), "bytes": sum(x["bytes"] for x in files), "sha256": sha(manifest.read_bytes())}))


if __name__ == "__main__":
    main()
