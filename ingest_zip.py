#!/usr/bin/env python3
"""Bulk-ingest a zip of text files into the running API via POST /ingest.

Usage:
  python ingest_zip.py northwind-sample-docs.zip
  python ingest_zip.py northwind-sample-docs.zip --query "What is the expense policy?"
"""

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import httpx


def find_text_files(root: Path, extensions: set[str]) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--extensions", default=".txt,.md")
    parser.add_argument("--query", help="Optional: after ingesting, run this through /debug/retrieve")
    args = parser.parse_args()

    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.extensions.split(",")}

    if not args.zip_path.exists():
        print(f"FAIL: {args.zip_path} does not exist")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(args.zip_path) as zf:
            zf.extractall(tmp_path)

        files = find_text_files(tmp_path, extensions)
        if not files:
            print(f"FAIL: no files with extensions {extensions} found in {args.zip_path}")
            return 1

        print(f"Found {len(files)} file(s) to ingest.\n")
        succeeded, failed, total_chunks = 0, 0, 0

        with httpx.Client(base_url=args.base_url, timeout=60.0) as client:
            for path in files:
                rel_path = path.relative_to(tmp_path).as_posix()
                text = path.read_text(encoding="utf-8", errors="ignore")
                if not text.strip():
                    print(f"  skip  {rel_path} (empty)")
                    continue

                try:
                    response = client.post(
                        "/ingest",
                        json={"document_id": rel_path, "text": text, "source": path.name},
                    )
                except httpx.ConnectError:
                    print(f"FAIL: cannot reach {args.base_url} - is the API running?")
                    return 1

                if response.status_code == 200:
                    chunks = response.json()["chunks_indexed"]
                    total_chunks += chunks
                    succeeded += 1
                    print(f"  ok    {rel_path} -> {chunks} chunk(s)")
                else:
                    failed += 1
                    print(f"  fail  {rel_path} -> HTTP {response.status_code}: {response.text}")

            print(f"\n{succeeded} succeeded, {failed} failed, {total_chunks} chunks indexed this run.")

            health = client.get("/debug/pinecone").json()
            if health["status"] == "ok":
                print(
                    f"Vector store total: {health['total_vector_count']} chunks in "
                    f"index {health['index']!r} (Pinecone stats can lag a few seconds "
                    "behind the latest upsert)."
                )
            else:
                print(f"Could not read vector store total: {health['detail']}")

            if args.query:
                print(f"\nQuerying /debug/retrieve?q={args.query!r} ...")
                response = client.get("/debug/retrieve", params={"q": args.query})
                response.raise_for_status()
                for match in response.json()["matches"]:
                    preview = (match["text"] or "")[:100].replace("\n", " ")
                    print(f"  {match['score']:.3f}  {match['document_id']}  {preview}...")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
