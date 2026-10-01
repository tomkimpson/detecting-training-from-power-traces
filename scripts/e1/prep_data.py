"""Tokenise FineWeb-edu for the E1 learning-cost pilot (exploratory).

Reads one pinned parquet shard of FineWeb-edu ``sample-10BT``, tokenises it with
the GPT-2 BPE (tiktoken), prefixing each document with <|endoftext|>, and writes:

    <out>/val.bin    the first ``--val-tokens`` tokens (whole documents)
    <out>/train.bin  the next ``--train-tokens`` tokens (whole documents)

as uint16, plus results/e1/data_manifest.json (source, revision, counts, sha256).
Validation and training data are disjoint document ranges of the same shard.

Two steps, because compute nodes have no internet (set HF_HOME and
TIKTOKEN_CACHE_DIR as in scripts/slurm/e1_prep.sbatch for both):
    python scripts/e1/prep_data.py --download-only      # login node: fetch shard + BPE
    sbatch scripts/slurm/e1_prep.sbatch                 # compute node: tokenise
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATASET = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SHARD = "sample/10BT/000_00000.parquet"


def fetch() -> Path:
    """The pinned shard, plus the GPT-2 BPE files (tiktoken caches them under
    $TIKTOKEN_CACHE_DIR), both of which need the internet on first call."""
    import tiktoken
    from huggingface_hub import hf_hub_download
    tiktoken.get_encoding("gpt2")
    return Path(hf_hub_download(DATASET, SHARD, repo_type="dataset", revision=REVISION))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=Path("/fred/oz022/tkimpson/e1_data"))
    ap.add_argument("--val-tokens", type=float, default=5e6)
    ap.add_argument("--train-tokens", type=float, default=500e6)
    ap.add_argument("--download-only", action="store_true")
    a = ap.parse_args()

    shard = fetch()
    print(f"shard: {shard}")
    if a.download_only:
        return

    import pyarrow.parquet as pq
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    a.out.mkdir(parents=True, exist_ok=True)

    targets = {"val": int(a.val_tokens), "train": int(a.train_tokens)}
    files = {k: (a.out / f"{k}.bin").open("wb") for k in targets}
    counts = {k: 0 for k in targets}
    docs = {k: 0 for k in targets}
    split = "val"
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=4096, columns=["text"]):
        for ids in enc.encode_ordinary_batch(batch.column("text").to_pylist(),
                                             num_threads=threads):
            if counts[split] >= targets[split]:
                if split == "train":
                    break
                split = "train"
            arr = np.asarray([eot] + ids, dtype=np.uint16)
            files[split].write(arr.tobytes())
            counts[split] += arr.size
            docs[split] += 1
        if counts["train"] >= targets["train"]:
            break
        print(f"  val {counts['val']:.3g}  train {counts['train']:.3g}", flush=True)
    for f in files.values():
        f.close()

    manifest = dict(
        dataset=DATASET, revision=REVISION, shard=SHARD, tokenizer="tiktoken gpt2",
        tiktoken=tiktoken.__version__, document_prefix="<|endoftext|>", dtype="uint16",
        split_rule="val = first whole documents of the shard, train = the following ones",
        tokens=counts, documents=docs,
        sha256={k: sha256(a.out / f"{k}.bin") for k in targets},
        out_dir=str(a.out),
    )
    dest = REPO / "results" / "e1" / "data_manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
