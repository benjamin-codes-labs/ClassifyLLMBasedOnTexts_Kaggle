"""Commands: audit, cross-validate, finalize, and predict."""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits

from attribution import VERSION, Predictor, weighted_predictions, definitions
from dataio import (
    SEED, digest_file, embedding_rows, embedding_spec, read_competition,
    split_prompts, submit, validate_text,
)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def audit(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train, test, labels = read_competition(args.data)
    sample = pd.read_parquet(Path(args.data)/"sample_submission.parquet")
    report = {"labels": labels, "files": {}, "class_counts": train.model.value_counts().to_dict(),
              "prompt_sizes": {str(k): int(v) for k,v in train.groupby("prompt_id").size().value_counts().items()},
              "sample_submission": sample.to_dict("records")}
    for name, frame in [("train", train), ("test", test)]:
        words = frame.text.str.split().str.len()
        report["files"][name] = {
            "rows": len(frame), "columns": frame.columns.tolist(),
            "sha256": digest_file(Path(args.data)/(name+".csv")),
            "missing": frame.isna().sum().to_dict(),
            "duplicate_ids": int(frame.id.duplicated().sum()),
            "duplicate_texts": int(frame.text.duplicated().sum()),
            "word_count": words.describe(percentiles=[.05,.25,.5,.75,.95]).to_dict(),
        }
    report["train_test_exact_text_overlap"] = len(set(train.text)&set(test.text))
    report["sample_ids_match_visible_test"] = sample.id.tolist() == test.id.tolist()
    save_json(output/"audit.json", report)
    print(json.dumps(report, indent=2))


def blend_candidates(names, single_scores):
    for name in names:
        yield {name: 1.0}
    for first, second in itertools.combinations(names, 2):
        if first.startswith("neural_") and second.startswith("neural_"):
            continue
        for weight in [.25, .5, .75]:
            yield {first: weight, second: 1-weight}
    has_encoder=any(name.startswith(("encoder_","prompt_","neural_")) for name in names)
    pool=names if has_encoder else sorted(names,key=single_scores.get,reverse=True)[:4]
    for trio in itertools.combinations(pool, 3):
        if sum(name.startswith("neural_") for name in trio)>1:
            continue
        if has_encoder and not (
            any(name.startswith(("lexical_","structure_")) for name in trio)
            and any(name.startswith(("encoder_","prompt_","neural_")) for name in trio)
        ):
            continue
        yield dict(zip(trio, [1/3]*3))
        for dominant in trio:
            yield {name: .5 if name == dominant else .25 for name in trio}


def choose(members, y, folds):
    single = {name: float(balanced_accuracy_score(y, p.argmax(1))) for name,p in members.items()}
    candidates = []
    for weights in blend_candidates(list(members), single):
        p = weighted_predictions(members, weights)
        score = float(balanced_accuracy_score(y, p.argmax(1)))
        fold_scores = [float(balanced_accuracy_score(y[folds==f], p[folds==f].argmax(1)))
                       for f in sorted(set(folds))]
        candidates.append({"weights": weights, "score": score, "fold_scores": fold_scores})
    candidates.sort(key=lambda c: c["score"], reverse=True)
    # Prefer a simpler ensemble when the gain is under 0.2 percentage points.
    near_best = [c for c in candidates if c["score"] >= candidates[0]["score"]-.002]
    selected = min(near_best, key=lambda c: (len(c["weights"]), -c["score"]))
    return selected, single, candidates


def cross_validate(args):
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=True)
    train, _, labels = read_competition(args.data)
    assigned = split_prompts(train, run/"splits.csv")
    target = pd.Categorical(train.model, categories=labels).codes
    vectors = embedding_rows(args.embeddings, train)
    specification = {
        "version": VERSION, "seed": SEED, "train_sha256": digest_file(Path(args.data)/"train.csv"),
        "encoder": embedding_spec(args.embeddings), "labels": labels,
    }
    if (run/"specification.json").exists():
        if json.loads((run/"specification.json").read_text()) != specification:
            raise ValueError("Output directory contains a different experiment")
    save_json(run/"specification.json", specification)
    oof = {}
    elapsed = []
    for fold in range(5):
        fit_idx = np.flatnonzero((assigned.fold >= 0) & (assigned.fold != fold))
        val_idx = np.flatnonzero(assigned.fold == fold)
        assert not set(train.prompt_id.iloc[fit_idx]) & set(train.prompt_id.iloc[val_idx])
        saved = run/f"fold_{fold}.npz"
        started = time.perf_counter()
        if saved.exists():
            with np.load(saved, allow_pickle=False) as f:
                if not np.array_equal(f["rows"], val_idx):
                    raise ValueError("Cached validation rows changed")
                prediction = {name:f[name] for name in f.files if name != "rows"}
        else:
            prediction = {}
        missing = set(definitions(vectors is not None))-set(prediction)
        if missing:
            print(f"Fold {fold}: {len(fit_idx)} train; {len(val_idx)} validation", flush=True)
            model = Predictor(labels,selected=missing).fit(
                train.text.iloc[fit_idx].tolist(), target[fit_idx],
                None if vectors is None else vectors[fit_idx],train.prompt_id.iloc[fit_idx].to_numpy())
            prediction.update(model.predict_members(
                train.text.iloc[val_idx].tolist(), None if vectors is None else vectors[val_idx]))
            np.savez_compressed(saved, rows=val_idx, **prediction)
        elapsed.append(time.perf_counter()-started)
        for name,p in prediction.items():
            oof.setdefault(name, np.zeros((len(train),len(labels)),dtype=np.float32))[val_idx] = p
        print("Fold scores:", {name: round(balanced_accuracy_score(target[val_idx],p.argmax(1)),4)
                              for name,p in prediction.items()}, flush=True)
    dev = np.flatnonzero(assigned.fold >= 0)
    members = {name:p[dev] for name,p in oof.items()}
    choice, single, candidates = choose(members, target[dev], assigned.fold.to_numpy()[dev])
    selection = {
        "specification": specification, "development_rows": len(dev),
        "holdout_rows": int((assigned.fold == -1).sum()),
        "component_scores": single, "selected": choice,
        "candidate_count": len(candidates), "top_candidates": candidates[:10],
        "fold_seconds": elapsed,
        "note": "Tuned development CV. Holdout has not been evaluated by this command.",
    }
    save_json(run/"selection.json", selection)
    np.savez_compressed(run/"oof.npz", rows=dev, y=target[dev],
                        folds=assigned.fold.to_numpy()[dev], **members)
    print(json.dumps(selection, indent=2), flush=True)


def evaluation(y, p, prompts, labels):
    predicted = p.argmax(1)
    matrix = confusion_matrix(y, predicted, labels=np.arange(len(labels)))
    groups = [np.flatnonzero(prompts==g) for g in sorted(set(prompts))]
    rng = np.random.default_rng(SEED)
    resampled = []
    for _ in range(1000):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
        resampled.append(balanced_accuracy_score(y[idx],predicted[idx]))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y,predicted)),
        "prompt_bootstrap_95pct": np.quantile(resampled,[.025,.975]).tolist(),
        "recall_by_model": dict(zip(labels,(matrix.diagonal()/matrix.sum(1)).tolist())),
        "confusion_matrix": matrix.tolist(),
    }


def combine(args):
    """Select using matching OOF rows; never read holdout targets here."""
    base, neural, output = Path(args.base), Path(args.neural), Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    selection = json.loads((base/"selection.json").read_text())
    neural_spec = json.loads((neural/"specification.json").read_text())
    if selection["specification"]["train_sha256"] != neural_spec["train_sha256"]:
        raise ValueError("Experiments use different training data")
    if selection["specification"]["encoder"]["checkpoint"] != neural_spec["checkpoint"]:
        raise ValueError("Experiments use different pretrained checkpoints")
    with np.load(base/"oof.npz",allow_pickle=False) as a, np.load(neural/"oof.npz",allow_pickle=False) as b:
        for key in ["rows","y","folds"]:
            if not np.array_equal(a[key],b[key]):
                raise ValueError("Experiments use different validation splits")
        members = {k:a[k] for k in a.files if k not in {"rows","y","folds"}}
        members.update({k:b[k] for k in b.files if k not in {"rows","y","folds"}})
        choice,single,candidates=choose(members,a["y"],a["folds"])
        np.savez_compressed(output/"oof.npz",rows=a["rows"],y=a["y"],folds=a["folds"],**members)
    selection.update(selected=choice,component_scores=single,candidate_count=len(candidates),
                     top_candidates=candidates[:10])
    selection["specification"]["neural_recipe"]=neural_spec["recipe"]
    save_json(output/"selection.json",selection)
    (output/"splits.csv").write_bytes((base/"splits.csv").read_bytes())
    print("Combined selection:",json.dumps(choice,indent=2),flush=True)


def finalize(args):
    run, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((run/"selection.json").read_text())
    spec = selection["specification"]
    if spec["version"] != VERSION or spec["encoder"] != embedding_spec(args.embeddings):
        raise ValueError("Final fit differs from validated implementation or encoder")
    if "neural_recipe" in spec:
        from neural import RECIPE
        if spec["neural_recipe"] != RECIPE:
            raise ValueError("Neural recipe differs from validation")
    if spec["train_sha256"] != digest_file(Path(args.data)/"train.csv"):
        raise ValueError("Training data changed after model selection")
    train,test,labels = read_competition(args.data)
    if labels != spec["labels"]:
        raise ValueError("Label ordering changed")
    split = split_prompts(train,run/"splits.csv")
    y = pd.Categorical(train.model,categories=labels).codes
    vectors = embedding_rows(args.embeddings,train)
    test_vectors = embedding_rows(args.embeddings,test)
    weights = selection["selected"]["weights"]
    dev, held = np.flatnonzero(split.fold>=0), np.flatnonzero(split.fold<0)
    held_path = output/"holdout.json"
    if held_path.exists():
        held_report = json.loads(held_path.read_text())
        if held_report["weights"] != weights or held_report["specification"] != spec:
            raise ValueError("Holdout output exists for a different selection")
    else:
        print("Evaluating the fixed selection on 30 held-out prompts.", flush=True)
        model = Predictor(labels,selected=weights).fit(
            train.text.iloc[dev].tolist(),y[dev],None if vectors is None else vectors[dev],
            train.prompt_id.iloc[dev].to_numpy())
        p = weighted_predictions(model.predict_members(
            train.text.iloc[held].tolist(),None if vectors is None else vectors[held]),weights)
        held_report = evaluation(y[held],p,train.prompt_id.iloc[held].to_numpy(),labels)
        held_report.update(weights=weights,specification=spec)
        save_json(held_path,held_report)
        pd.DataFrame({"id":train.id.iloc[held].to_numpy(),"truth":train.model.iloc[held].to_numpy(),
                      "prediction":np.asarray(labels)[p.argmax(1)]}).to_csv(output/"holdout_predictions.csv",index=False)
        print("Holdout balanced accuracy:",held_report["balanced_accuracy"],flush=True)
        del model
    print("Refitting selected components on all training rows.",flush=True)
    started=time.perf_counter()
    model=Predictor(labels,selected=weights).fit(train.text.tolist(),y,vectors,train.prompt_id.to_numpy())
    p=weighted_predictions(model.predict_members(test.text.tolist(),test_vectors),weights)
    bundle={"predictor":model,"weights":weights,"labels":labels,"specification":spec,
            "versions":{"python":platform.python_version(),"sklearn":sklearn.__version__,"numpy":np.__version__}}
    joblib.dump(bundle,output/"predictor.joblib",compress=3)
    result=submit(test,p,labels,output/"submission.parquet")
    pd.DataFrame(p,index=test.id,columns=labels).to_csv(output/"probabilities.csv")
    save_json(output/"selection.json",selection)
    save_json(output/"runtime.json",{"final_fit_and_sample_prediction_seconds":time.perf_counter()-started})
    print(result.to_string(index=False),flush=True)


def predict(args):
    bundle=joblib.load(args.model)
    if bundle["specification"]["version"] != VERSION:
        raise ValueError("Saved predictor version mismatch")
    if bundle["specification"]["encoder"] != embedding_spec(args.embeddings):
        raise ValueError("Encoder configuration mismatch")
    test=pd.read_csv(args.test,dtype=str)
    validate_text(test)
    vectors=embedding_rows(args.embeddings,test)
    p=weighted_predictions(bundle["predictor"].predict_members(test.text.tolist(),vectors),bundle["weights"])
    print(submit(test,p,bundle["labels"],args.output).to_string(index=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest="command",required=True)
    for name,func in [("audit",audit),("cv",cross_validate),("finalize",finalize)]:
        sub=commands.add_parser(name)
        sub.add_argument("--data",default="Data")
        sub.add_argument("--output",required=True)
        if name != "audit":
            sub.add_argument("--embeddings")
        if name == "finalize":
            sub.add_argument("--run",required=True)
        sub.set_defaults(func=func)
    sub=commands.add_parser("predict")
    sub.add_argument("--model",required=True)
    sub.add_argument("--test",default="Data/test.csv")
    sub.add_argument("--embeddings")
    sub.add_argument("--output",default="submission.parquet")
    sub.set_defaults(func=predict)
    sub=commands.add_parser("combine")
    sub.add_argument("--base",required=True)
    sub.add_argument("--neural",required=True)
    sub.add_argument("--output",required=True)
    sub.set_defaults(func=combine)
    args=p.parse_args()
    with threadpool_limits(limits=4):
        args.func(args)


if __name__=="__main__":
    main()
