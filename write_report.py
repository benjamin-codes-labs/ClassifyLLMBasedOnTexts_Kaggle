"""Write the final evidence report from actual saved experiment artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def create(best_run,best_delivery,destination):
    best_run,best_delivery=Path(best_run),Path(best_delivery)
    selection=json.loads((best_run/"selection.json").read_text())
    held=json.loads((best_delivery/"holdout.json").read_text())
    cpu=json.loads(Path("deliverables/cpu/holdout.json").read_text())
    baseline=json.loads(Path("runs/classical/selection.json").read_text())
    labels=selection["specification"]["labels"]
    matrix=np.array(held["confusion_matrix"])
    confidence=held["prompt_bootstrap_95pct"]
    lines=[
        "# Project 2 — results and delivery report",
        "",
        "This project was implemented afresh in `Desktop/KaggleProject2`, using",
        "the original competition data and measured experiments. No research",
        "papers, competitor code, or artifacts from the first project were used.",
        "",
        "## Measured results",
        "",
        "| Pipeline | Development balanced accuracy | Holdout balanced accuracy |",
        "| --- | ---: | ---: |",
        f"| Classical CPU ensemble | {baseline['selected']['score']:.4f} | {cpu['balanced_accuracy']:.4f} |",
        f"| Selected final pipeline | {selection['selected']['score']:.4f} | {held['balanced_accuracy']:.4f} |",
        "",
        f"The final holdout interval from 1,000 prompt-block bootstrap resamples is",
        f"**{confidence[0]:.4f}–{confidence[1]:.4f}**. This measures uncertainty on",
        "the reserved prompts; it does not account for an unknown shift in hidden",
        "Kaggle data. Uniform random guessing across 18 classes has expected",
        "balanced accuracy 0.0556.",
        "",
        f"The selected run is `{best_run}`. It uses these weights:",
        "",
        "| Component | Weight |",
        "| --- | ---: |",
    ]
    lines += [f"| `{name}` | {weight:.4f} |" for name,weight in selection["selected"]["weights"].items()]
    lines += [
        "",
        "## Component ablations",
        "",
        "These are out-of-fold development measurements on the same prompt split.",
        "They are not Kaggle leaderboard results. Ensemble selection used these",
        "predictions, so its development score is a tuned estimate.",
        "",
        "| Component | Balanced accuracy |",
        "| --- | ---: |",
    ]
    lines += [f"| `{name}` | {score:.4f} |" for name,score in sorted(
        selection["component_scores"].items(),key=lambda x:x[1],reverse=True)]
    lines += [
        "",
        "## Validation and decisions",
        "",
        "- Training contains 2,699 passages, 18 model labels, and 150 prompts.",
        "- The fixed split uses 120 development prompts and 30 holdout prompts.",
        f"- There are {selection['development_rows']} development rows and "
        f"{selection['holdout_rows']} holdout rows.",
        "- Five validation folds keep all passages from each prompt together.",
        "- All vocabulary fitting, scalers, supervised models, and prompt-based",
        "  regularization are fitted only on the current training fold.",
        "- Frozen encoder extraction is deterministic per passage and uses no",
        "  competition labels. The exact checkpoint revision is recorded.",
        "- Ensembles use nonnegative weights from a bounded candidate search;",
        "  a simpler candidate is preferred within 0.002 of the best CV score.",
        f"- The final search evaluated {selection['candidate_count']} candidate weight combinations.",
        "- The final selection was fixed before its holdout evaluation.",
        "- The final predictor was then refitted on all 2,699 training examples.",
        "- Each prediction is independent of test batch composition. No class",
        "  quotas, prompt matching, or test labels are used.",
        "",
        "The holdout was excluded from this project's fitting and model selection.",
        "The dataset had already been inspected descriptively. The CPU baseline",
        "and final pipeline share this holdout; their results are paired",
        "comparisons, not independent replications.",
        "",
        "## Fine-tuning pilot",
        "",
        "Two of five planned neural folds completed. Their epoch-three balanced",
        "accuracies were 0.3519 and 0.3668, versus 0.4005 and 0.3940 for the",
        "frozen encoder logistic model on those same folds. The machine",
        "experienced substantial memory pressure during training, so this",
        "candidate was stopped. Its incomplete results were excluded from",
        "ensemble selection. `runs/neural/pilot_summary.json` records the",
        "completed-fold measurements; no five-fold neural score is claimed.",
        "",
        "## Holdout recall by exact model",
        "",
        "| Model | Recall |",
        "| --- | ---: |",
    ]
    lines += [f"| `{name}` | {recall:.4f} |" for name,recall in held["recall_by_model"].items()]
    off_diagonal=matrix.copy()
    np.fill_diagonal(off_diagonal,0)
    worst=sorted(((int(off_diagonal[i,j]),i,j) for i in range(len(labels)) for j in range(len(labels))
                  if i!=j and off_diagonal[i,j]>0),reverse=True)[:10]
    lines += ["","## Largest holdout confusions","","| True model | Predicted model | Passages |",
              "| --- | --- | ---: |"]
    lines += [f"| `{labels[i]}` | `{labels[j]}` | {count} |" for count,i,j in worst]
    verification_path=Path("deliverables/verification.json")
    if verification_path.exists():
        verification=json.loads(verification_path.read_text())
        lines += ["","## Executed verification",""]
        lines += [f"- {message}" for message in verification["checks"]]
    lines += [
        "",
        "## Submission files",
        "",
        "`submission.parquet` is the selected pipeline's output for the three",
        "visible test rows. `submission_cpu.parquet` is the simpler CPU fallback.",
        "Both contain exactly `id` and `model`, preserve the input IDs, have no",
        "nulls, and use valid class names.",
        "",
        "| ID | Selected prediction |",
        "| --- | --- |",
    ]
    sample=pd.read_parquet(best_delivery/"submission.parquet")
    lines += [f"| `{row.id}` | `{row.model}` |" for row in sample.itertuples()]
    lines += [
        "",
        "The visible test labels are unknown. These three predictions cannot",
        "establish accuracy or rank. `kaggle_submission.ipynb` processes the",
        "actual runtime test file and writes the required output on Kaggle.",
        "The pretrained input archive is `deliverables/kaggle_encoder_input.zip`.",
        "",
        "## Reproducibility and practical limits",
        "",
        "- Source code, the dependency lock, split assignments, OOF predictions,",
        "  model selections, and fitted predictors are saved in this project.",
        "- Unit tests cover prompt separation, independent batch prediction,",
        "  encoder text checksums, and the Parquet output contract.",
        "- The CPU notebook requires no external checkpoint.",
        "- The enhanced notebook requires the pinned ELECTRA checkpoint as an",
        "  attached input and runs with internet disabled.",
        "- The local machine is an Apple M2 Pro with 16 GB unified memory.",
        "- No paid API, synthetic training corpus, or hidden-test labels were used.",
        "- No Kaggle scoring run was performed from this environment. There is",
        "  no measured leaderboard score and no evidence supporting a first-place claim.",
        "",
        "See `README.md` for commands and Kaggle input instructions. Improving",
        "the remaining sibling-model errors requires further independent",
        "experiments and actual leaderboard feedback; the report does not assign",
        "an invented score to that future work.",
    ]
    Path(destination).write_text("\n".join(lines)+"\n")
    pd.DataFrame(matrix,index=labels,columns=labels).to_csv(best_delivery/"confusion_matrix.csv")
    print("Wrote",destination)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    parser.add_argument("--delivery",default="deliverables/best")
    parser.add_argument("--output",default="RESULTS.md")
    args=parser.parse_args()
    create(args.run,args.delivery,args.output)
