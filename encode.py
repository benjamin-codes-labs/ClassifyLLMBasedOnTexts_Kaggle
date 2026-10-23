"""Local frozen-encoder features. Reads model metadata, not research papers."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dataio import text_hashes, validate_text

CHECKPOINT = "google/electra-large-discriminator"
ENCODING_VERSION = "electra-moments-1"


def fetch_checkpoint(folder):
    from huggingface_hub import model_info, snapshot_download
    info = model_info(CHECKPOINT)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        CHECKPOINT, revision=info.sha, local_dir=folder,
        allow_patterns=["config.json", "pytorch_model.bin", "tokenizer.json",
                        "tokenizer_config.json", "vocab.txt", "README.md", "LICENSE*"],
    )
    manifest = {"id": CHECKPOINT, "revision": info.sha, "license": info.card_data.license}
    (folder/"manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def encode_frame(frame, checkpoint, destination, batch_size=6, device="auto"):
    import torch
    from transformers import AutoModelForPreTraining, AutoTokenizer

    validate_text(frame)
    destination, checkpoint = Path(destination), Path(checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((checkpoint/"manifest.json").read_text())
    specification = {"version": ENCODING_VERSION, "checkpoint": manifest, "window_tokens": 510}
    spec_text = json.dumps(specification, sort_keys=True)
    hashes = dict(zip(frame.id, text_hashes(frame.text)))
    completed = {}
    if destination.exists():
        with np.load(destination, allow_pickle=False) as f:
            if str(f["specification"]) != spec_text:
                raise ValueError("Existing embeddings use a different encoder")
            for row_id, hashed, vector in zip(f["ids"], f["hashes"], f["vectors"]):
                if row_id in hashes:
                    if hashes[row_id] != hashed:
                        raise ValueError("Existing embedding text changed")
                    completed[str(row_id)] = vector
    todo = frame.loc[~frame.id.isin(completed)]
    if todo.empty:
        print("Encoder cache is complete.", flush=True)
        return
    torch.set_num_threads(4)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForPreTraining.from_pretrained(
        checkpoint, local_files_only=True,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    ).to(device).eval()
    if not hasattr(model, "electra"):
        raise ValueError("Expected a standard ELECTRA checkpoint")
    print(f"Encoder loaded on {device}; processing {len(todo)} passages", flush=True)
    work = []
    required, counts = {}, {}
    lengths = {}
    for row in todo.itertuples():
        tokens = tokenizer.encode(row.text, add_special_tokens=False, truncation=False)
        if not tokens:
            raise ValueError("Text has no tokens")
        chunks = [tokens[start:start+510] for start in range(0, len(tokens), 510)]
        required[row.id] = len(chunks)
        counts[row.id] = 0
        lengths[row.id] = (len(tokens), len(row.text))
        for chunk in chunks:
            work.append((row.id, chunk))
    # Sorting windows reduces padding; IDs restore the original correspondence.
    work.sort(key=lambda item: len(item[1]))
    moments, scores = {}, {}
    started = time.perf_counter()

    def checkpoint_cache():
        ordered = [row_id for row_id in frame.id if row_id in completed]
        temp = destination.with_suffix(".partial.npz")
        np.savez_compressed(
            temp, ids=np.asarray(ordered), hashes=np.asarray([hashes[i] for i in ordered]),
            vectors=np.stack([completed[i] for i in ordered]), specification=spec_text,
        )
        temp.replace(destination)

    def progress():
        elapsed = time.perf_counter()-started
        finished = len(completed)-(len(frame)-len(todo))
        remaining = len(frame)-len(completed)
        print(f"Encoded {len(completed)}/{len(frame)}; {elapsed:.0f}s; "
              f"ETA {remaining*elapsed/max(finished,1)/60:.1f}min", flush=True)

    last_saved = len(completed)
    with torch.inference_mode():
        for start in range(0, len(work), batch_size):
            batch = work[start:start+batch_size]
            ids = [[tokenizer.cls_token_id]+tokens+[tokenizer.sep_token_id] for _,tokens in batch]
            inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(device)
            encoded = model.electra(**inputs, output_hidden_states=True)
            states = encoded.hidden_states
            discriminator = model.discriminator_predictions(encoded.last_hidden_state)
            for position,(row_id,tokens) in enumerate(batch):
                n = len(tokens)
                last = states[-1][position,1:n+1].float()
                middle = states[len(states)//2][position,1:n+1].float()
                chunk_moments = np.stack([
                    middle.sum(0).cpu().numpy(), last.sum(0).cpu().numpy(),
                    last.square().sum(0).cpu().numpy(),
                    (states[-1][position,0].float()*n).cpu().numpy(),
                ])
                moments[row_id] = moments.get(row_id,0)+chunk_moments
                scores.setdefault(row_id,[]).append(discriminator[position,1:n+1].float().cpu().numpy())
                counts[row_id] += 1
                if counts[row_id] == required[row_id]:
                    total,char_count = lengths[row_id]
                    means = moments.pop(row_id)/total
                    deviations = np.sqrt(np.maximum(means[2]-means[1]**2,0))
                    logits = np.concatenate(scores.pop(row_id))
                    summary = np.array([
                        logits.mean(), logits.std(), logits.min(), logits.max(),
                        *np.quantile(logits,[.05,.25,.5,.75,.95]),
                        np.log1p(total), char_count/total, float(required[row_id]>1),
                    ])
                    vector = np.concatenate([means[0],means[1],deviations,means[3],summary]).astype(np.float32)
                    if not np.isfinite(vector).all():
                        raise ValueError(f"Nonfinite encoder output for {row_id}")
                    completed[row_id] = vector
            if len(completed)-last_saved >= 120 or start+batch_size >= len(work):
                checkpoint_cache()
                progress()
                last_saved = len(completed)
    print("Completed encoder cache:", destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/electra-large")
    parser.add_argument("--data", default="Data")
    parser.add_argument("--test", help="Encode a separate runtime test file instead of train plus visible test")
    parser.add_argument("--output", default="cache/electra.npz")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.download:
        fetch_checkpoint(args.checkpoint)
    if args.test:
        frame = pd.read_csv(args.test, dtype=str)[["id","text"]]
    else:
        frame = pd.concat([pd.read_csv(Path(args.data)/f"{name}.csv",dtype=str)[["id","text"]]
                           for name in ["train","test"]],ignore_index=True)
    encode_frame(frame,args.checkpoint,args.output,args.batch_size,args.device)


if __name__ == "__main__":
    main()
