"""Dataset contracts, immutable split assignment, and submission validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SEED = 71423


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_hashes(texts):
    return np.array([hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts])


def read_competition(directory):
    directory = Path(directory)
    train = pd.read_csv(directory/"train.csv", dtype=str)
    test = pd.read_csv(directory/"test.csv", dtype=str)
    expected = {"train": ["id", "prompt_id", "text", "model"], "test": ["id", "text"]}
    for name, frame in [("train", train), ("test", test)]:
        if frame.columns.tolist() != expected[name]:
            raise ValueError(f"{name}: unexpected columns")
        validate_text(frame)
        if frame.isna().any().any():
            raise ValueError(f"{name}: missing cell")
    labels = sorted(train.model.unique())
    if len(labels) != 18 or train.prompt_id.nunique() < 40:
        raise ValueError("Unexpected competition training data")
    if set(train.id) & set(test.id):
        raise ValueError("Train/test IDs overlap")
    sample = pd.read_parquet(directory/"sample_submission.parquet")
    if sample.columns.tolist() != ["id", "model"]:
        raise ValueError("Unexpected sample submission schema")
    # Only check the sample's format: hidden test files may have different IDs.
    return train, test, labels


def validate_text(frame):
    if not {"id", "text"}.issubset(frame.columns):
        raise ValueError("Input requires id and text")
    if frame[["id", "text"]].isna().any().any() or not frame.id.is_unique:
        raise ValueError("Missing text/ID or duplicate IDs")
    if frame.id.str.strip().eq("").any() or frame.text.str.strip().eq("").any():
        raise ValueError("Empty text/ID")


def split_prompts(train, destination):
    """30 holdout prompts; five folds across all remaining prompts."""
    destination = Path(destination)
    if destination.exists():
        saved = pd.read_csv(destination, dtype={"id": str, "prompt_id": str})
        if not saved[["id", "prompt_id"]].equals(train[["id", "prompt_id"]]):
            raise ValueError("Saved split belongs to different data")
        if saved.groupby("prompt_id").fold.nunique().max() != 1:
            raise ValueError("A prompt crosses split boundaries")
        if set(saved.fold) != {-1, 0, 1, 2, 3, 4}:
            raise ValueError("Invalid fold assignments")
        if saved.loc[saved.fold == -1, "prompt_id"].nunique() != 30:
            raise ValueError("Expected 30 holdout prompts")
        return saved
    ordered = sorted(train.prompt_id.unique(),
                     key=lambda g: hashlib.sha256(f"{SEED}:{g}".encode()).hexdigest())
    held = set(ordered[:30])
    assigned = train[["id", "prompt_id"]].copy()
    assigned["fold"] = -1
    indices = np.flatnonzero(~train.prompt_id.isin(held))
    cv = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, val) in enumerate(cv.split(indices, groups=train.prompt_id.iloc[indices])):
        assigned.loc[indices[val], "fold"] = fold
    destination.parent.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(destination, index=False)
    return assigned


def embedding_rows(cache, frame):
    if cache is None:
        return None
    with np.load(cache, allow_pickle=False) as f:
        lookup = {row_id: i for i, row_id in enumerate(f["ids"])}
        if len(lookup) != len(f["ids"]):
            raise ValueError("Duplicate embedding IDs")
        positions = [lookup[row_id] for row_id in frame.id]
        if not np.array_equal(f["hashes"][positions], text_hashes(frame.text)):
            raise ValueError("Embedding text checksum mismatch")
        result = f["vectors"][positions]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite embeddings")
    return result


def embedding_spec(cache):
    if cache is None:
        return None
    with np.load(cache, allow_pickle=False) as f:
        return json.loads(str(f["specification"]))


def submit(test, probabilities, labels, path):
    validate_text(test)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(test), len(labels)):
        raise ValueError("Prediction shape mismatch")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Invalid prediction values")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Prediction rows must sum to one")
    result = pd.DataFrame({
        "id": test.id.to_numpy(),
        "model": np.asarray(labels)[probabilities.argmax(axis=1)],
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(path), result)
    return result
