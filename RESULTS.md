# Project 2 — results and delivery report

This project was implemented afresh in `Desktop/KaggleProject2`, using
the original competition data and measured experiments. No research
papers, competitor code, or artifacts from the first project were used.

## Measured results

| Pipeline | Development balanced accuracy | Holdout balanced accuracy |
| --- | ---: | ---: |
| Classical CPU ensemble | 0.3797 | 0.3630 |
| Selected final pipeline | 0.4510 | 0.4519 |

The final holdout interval from 1,000 prompt-block bootstrap resamples is
**0.4092–0.4963**. This measures uncertainty on
the reserved prompts; it does not account for an unknown shift in hidden
Kaggle data. Uniform random guessing across 18 classes has expected
balanced accuracy 0.0556.

The selected run is `runs/encoder`. It uses these weights:

| Component | Weight |
| --- | ---: |
| `structure_svm` | 0.2500 |
| `style_boost` | 0.2500 |
| `encoder_logistic_strong` | 0.5000 |

## Component ablations

These are out-of-fold development measurements on the same prompt split.
They are not Kaggle leaderboard results. Ensemble selection used these
predictions, so its development score is a tuned estimate.

| Component | Balanced accuracy |
| --- | ---: |
| `encoder_logistic_strong` | 0.4210 |
| `encoder_logistic` | 0.4200 |
| `encoder_logistic_light` | 0.4149 |
| `encoder_regularized` | 0.3704 |
| `prompt_regularized` | 0.3695 |
| `structure_logistic` | 0.3663 |
| `structure_svm` | 0.3662 |
| `prompt_regularized_light` | 0.3172 |
| `lexical_svm` | 0.3148 |
| `style_boost` | 0.3103 |
| `style_forest` | 0.2797 |
| `encoder_ridge` | 0.2672 |

## Validation and decisions

- Training contains 2,699 passages, 18 model labels, and 150 prompts.
- The fixed split uses 120 development prompts and 30 holdout prompts.
- There are 2159 development rows and 540 holdout rows.
- Five validation folds keep all passages from each prompt together.
- All vocabulary fitting, scalers, supervised models, and prompt-based
  regularization are fitted only on the current training fold.
- Frozen encoder extraction is deterministic per passage and uses no
  competition labels. The exact checkpoint revision is recorded.
- Ensembles use nonnegative weights from a bounded candidate search;
  a simpler candidate is preferred within 0.002 of the best CV score.
- The final search evaluated 714 candidate weight combinations.
- The final selection was fixed before its holdout evaluation.
- The final predictor was then refitted on all 2,699 training examples.
- Each prediction is independent of test batch composition. No class
  quotas, prompt matching, or test labels are used.

The holdout was excluded from this project's fitting and model selection.
The dataset had already been inspected descriptively. The CPU baseline
and final pipeline share this holdout; their results are paired
comparisons, not independent replications.

## Fine-tuning pilot

Two of five planned neural folds completed. Their epoch-three balanced
accuracies were 0.3519 and 0.3668, versus 0.4005 and 0.3940 for the
frozen encoder logistic model on those same folds. The machine
experienced substantial memory pressure during training, so this
candidate was stopped. Its incomplete results were excluded from
ensemble selection. `runs/neural/pilot_summary.json` records the
completed-fold measurements; no five-fold neural score is claimed.

## Holdout recall by exact model

| Model | Recall |
| --- | ---: |
| `claude-fable-5` | 0.3333 |
| `claude-opus-4-6` | 0.5667 |
| `claude-opus-4-8` | 0.4000 |
| `claude-opus-5` | 0.7667 |
| `claude-sonnet-5` | 0.3667 |
| `gemini-3.1-pro-preview` | 0.7667 |
| `gemini-3.6-flash` | 0.5333 |
| `gemini-3.7-flash` | 0.4000 |
| `gemini-3.8-flash` | 0.4667 |
| `glm-5.3` | 0.2333 |
| `glm-5.3-flash` | 0.1000 |
| `gpt-5.5` | 0.3667 |
| `gpt-5.6-luna` | 0.5333 |
| `gpt-5.6-sol` | 0.2667 |
| `gpt-5.6-terra` | 0.3333 |
| `gpt-6-astra` | 0.7667 |
| `kimi-k3` | 0.5000 |
| `qwen3.8-max` | 0.4333 |

## Largest holdout confusions

| True model | Predicted model | Passages |
| --- | --- | ---: |
| `gemini-3.7-flash` | `gemini-3.8-flash` | 11 |
| `gpt-5.6-sol` | `gpt-5.6-luna` | 8 |
| `gpt-5.6-terra` | `gpt-5.6-luna` | 7 |
| `gpt-5.6-sol` | `gpt-5.5` | 7 |
| `glm-5.3` | `claude-opus-5` | 7 |
| `gpt-5.6-terra` | `gpt-5.5` | 6 |
| `glm-5.3-flash` | `claude-fable-5` | 6 |
| `gemini-3.8-flash` | `gemini-3.7-flash` | 6 |
| `gemini-3.6-flash` | `gemini-3.1-pro-preview` | 6 |
| `gpt-5.5` | `kimi-k3` | 5 |

## Executed verification

- Seven automated contract and leakage-boundary tests passed.
- Both generated notebooks executed end to end locally with offline model-loading settings.
- Both notebook submission files exactly matched their saved-predictor submission files.
- The selected predictor was serialized, reloaded, and successfully used for inference.
- Reordering the visible test batch preserved selected-model probabilities within 1e-6.
- Both root Parquet files passed schema, ID, null, and allowed-label checks.
- Attribution inference on 4,000 rows took 15.1 seconds using precomputed encoder vectors.
- The enhanced notebook, including fresh encoder extraction and retraining, took 9.7 minutes locally.

## Submission files

`submission.parquet` is the selected pipeline's output for the three
visible test rows. `submission_cpu.parquet` is the simpler CPU fallback.
Both contain exactly `id` and `model`, preserve the input IDs, have no
nulls, and use valid class names.

| ID | Selected prediction |
| --- | --- |
| `00353fb1c541` | `claude-opus-4-8` |
| `005f5a9a6139` | `gpt-5.6-terra` |
| `0062a53d5487` | `gemini-3.7-flash` |

The visible test labels are unknown. These three predictions cannot
establish accuracy or rank. `kaggle_submission.ipynb` processes the
actual runtime test file and writes the required output on Kaggle.
The pretrained input archive is `deliverables/kaggle_encoder_input.zip`.

## Reproducibility and practical limits

- Source code, the dependency lock, split assignments, OOF predictions,
  model selections, and fitted predictors are saved in this project.
- Unit tests cover prompt separation, independent batch prediction,
  encoder text checksums, and the Parquet output contract.
- The CPU notebook requires no external checkpoint.
- The enhanced notebook requires the pinned ELECTRA checkpoint as an
  attached input and runs with internet disabled.
- The local machine is an Apple M2 Pro with 16 GB unified memory.
- No paid API, synthetic training corpus, or hidden-test labels were used.
- No Kaggle scoring run was performed from this environment. There is
  no measured leaderboard score and no evidence supporting a first-place claim.

See `README.md` for commands and Kaggle input instructions. Improving
the remaining sibling-model errors requires further independent
experiments and actual leaderboard feedback; the report does not assign
an invented score to that future work.
