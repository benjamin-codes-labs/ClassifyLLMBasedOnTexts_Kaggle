# Name That LLM — Stylometric and Neural Attribution

**An end-to-end NLP system for identifying the language model behind a
passage, combining interpretable writing-style features with pretrained
representations and validation on unseen prompts.**

Built for Kaggle's [Name That LLM competition](https://www.kaggle.com/competitions/name-that-llm),
this project addresses **18-class model attribution** across six model
providers. Its final ensemble reaches **45.19% balanced accuracy on a
540-passage holdout**, compared with **36.30%** for the classical baseline:
an **8.89 percentage-point improvement** on the same reserved prompts.

The implementation covers the full competition workflow: input auditing,
feature extraction, grouped cross-validation, component ablations, ensemble
selection, uncertainty estimates, model serialization, and offline notebook
submission.

## Competition and prediction task

Given a short English passage, predict the **exact generating model**.
Distinguishing providers alone is insufficient: separate versions within
the Claude, Gemini, GPT, and GLM families are individual target classes.
The remaining candidates include Kimi and Qwen.

| File | Rows | Fields | Purpose |
| --- | ---: | --- | --- |
| `train.csv` | 2,699 | `id`, `prompt_id`, `text`, `model` | Supervised training data |
| `test.csv` | 3 visible examples | `id`, `text` | Inference interface; replaced during scoring |
| `sample_submission.parquet` | 3 | `id`, `model` | Submission schema |

Training contains **150 prompts**, with almost one completion per model
per prompt. This creates a critical validation requirement: completions
from the same prompt must stay together when splitting the data.

The competition metric is **balanced accuracy**, the mean recall across
the 18 models:

```text
balanced_accuracy = mean(correct_predictions_for_class / examples_in_class)
```

Every class contributes equally. Uniform random guessing has expected
balanced accuracy **5.56%**. The local test examples have no ground-truth
labels; Kaggle evaluates the notebook on hidden passages.

## Dataset access and local file structure

**I do not publish or redistribute the competition dataset in this
repository.** Download the original files directly from the
[official Kaggle competition data page](https://www.kaggle.com/competitions/name-that-llm/data)
after signing in and accepting the applicable competition rules.

The `Data/` directory is excluded by `.gitignore` and may not exist in a
fresh checkout. Create it locally and extract the downloaded files into
the following layout, preserving the filenames and capital `D`:

```text
KaggleProject2/
├── README.md
├── workflow.py
├── dataio.py
├── ...
└── Data/                          # Local only; not published
    ├── train.csv                  # Labeled passages
    ├── test.csv                   # Unlabeled passages
    └── sample_submission.parquet  # Example output schema
```

The file counts in the competition table above describe the dataset
version used for the reported experiments. The downloaded test file is
only a small example; the notebook reads the actual test file supplied
by Kaggle at scoring time.

### Column definitions

| Column | Logical type | Meaning | Where it appears |
| --- | --- | --- | --- |
| `id` | String | Unique passage identifier; preserve leading zeros | All three files |
| `prompt_id` | String | Groups completions produced in response to the same prompt | Training only |
| `text` | String | Passage to classify, preserving its original punctuation and Unicode characters | Training and test |
| `model` | String | Exact generating-model label in training, or predicted label in a submission | Training and sample submission |

The classifier learns from `text` and the training target `model`.
`prompt_id` supports grouped validation, while `id` associates predictions
with the correct records. Neither identifier is treated as a text feature.

### Illustrative file contents

**All rows below are synthetic documentation examples, not excerpts from
the competition dataset.** The IDs, passages, and prompt assignments are
invented. The model names are valid competition labels, but their example
assignments do not claim that those models generated these sentences.
These shortened examples illustrate the schemas only; they cannot replace
the original data or reproduce the reported results.

**`Data/train.csv` — labeled examples**

```csv
id,prompt_id,text,model
000000000001,p000,"A brass compass rested beside the unopened map.",gpt-5.5
000000000002,p000,"Beside the folded map, a brass compass waited.",claude-opus-5
```

The repeated `prompt_id` illustrates two completions associated with one
prompt. Both belong in the same validation fold.

**`Data/test.csv` — passages requiring predictions**

```csv
id,text
000000000003,"Lantern light reflected in the window."
000000000004,"Thin clouds drifted above a quiet harbor."
```

The test schema contains neither `model` nor `prompt_id`. Prediction code
must work from the passage text without those fields.

**`Data/sample_submission.parquet` — output-format example**

Parquet is a binary table format. The following table illustrates its
contents after loading; it is not a CSV file to rename as `.parquet`.

| id | model |
| --- | --- |
| `000000000003` | `gpt-5.5` |
| `000000000004` | `gpt-5.5` |

The repeated model values illustrate template placeholders, not ground
truth for the test passages. A generated submission must contain exactly
`id` and `model`, with one valid model label per runtime test ID, no duplicate
IDs, and no missing values.

### Inspect the downloaded files

After downloading the real dataset, run this from the project root:

```python
import pandas as pd

train = pd.read_csv("Data/train.csv", dtype=str)
test = pd.read_csv("Data/test.csv", dtype=str)
sample_submission = pd.read_parquet("Data/sample_submission.parquet")

print(train.columns.tolist())              # ['id', 'prompt_id', 'text', 'model']
print(test.columns.tolist())               # ['id', 'text']
print(sample_submission.columns.tolist())  # ['id', 'model']
```

Generated prediction files such as `submission.parquet` are outputs
separate from the input files under `Data/`.

## Results

| Model / pipeline | Grouped development CV | Reserved-prompt holdout |
| --- | ---: | ---: |
| Classical CPU ensemble | 37.97% | 36.30% |
| Selected stylometric + encoder ensemble | **45.10%** | **45.19%** |

The final holdout consists of **30 prompts and 540 passages**. Its
prompt-bootstrap 95% interval is **40.92%–49.63%**. Development scores were
used for model selection; the holdout was excluded from fitting and weight
selection in this project.

These are **local measurements**, not a Kaggle leaderboard score or a
claim of a competition-winning result. The dataset had been explored and
modeled in earlier project work; this split was excluded from Project 2's
fitting and weight selection. Full ablations, class recalls, and the confusion
analysis are available in [`RESULTS.md`](RESULTS.md).

## Technical architecture

```text
Passage
  ├── Character / word TF-IDF + grammatical outline + numeric features
  │       └── Linear SVM ────────────────────────────── 25%
  ├── Explicit lexical, sentence, and punctuation statistics
  │       └── CatBoost multiclass classifier ────────── 25%
  └── Frozen ELECTRA-large representations
          └── Regularized multinomial logistic model ─ 50%
                           │
                    Probability ensemble
                           │
                     Exact model label
```

**Text and structural branch.** Case-preserving character n-grams of length
2–6 capture punctuation and spelling patterns. Word unigrams/bigrams add
lexical information. A separate grammatical outline retains function words
and punctuation while replacing content words with capitalization and
length categories. Robustly scaled numeric features complement the sparse
representation.

**Stylometric branch.** CatBoost models nonlinear interactions among
vocabulary diversity, repetition, word-length distributions, sentence
structure, function-word frequencies, Unicode punctuation, and opening/
closing patterns. The model is trained directly on these explicit features.

**Neural branch.** A frozen `google/electra-large-discriminator` checkpoint
produces **4,108 features per passage**: intermediate-layer means,
final-layer means and standard deviations, the classification-token
representation, and token-discriminator statistics. A regularized
multinomial classifier combines these vectors with writing-style features.
The checkpoint is used as a representation extractor; it is not one of
the candidate authors.

**Ensemble selection.** Twelve components were evaluated with five-fold
grouped validation. A bounded search evaluated 714 nonnegative weight
combinations, preferring a simpler candidate when its score was within
0.002 of the best development result. The final three-component ensemble
was then refitted on all labeled passages.

## Engineering highlights

- **Prompt-aware evaluation:** five development folds across 120 prompts,
  plus a separate 30-prompt holdout.
- **Measured component contributions:** ablations include sparse linear
  models, boosted trees, randomized trees, encoder classifiers, and
  training-time prompt regularization.
- **Batch-independent inference:** predictions do not assume balanced test
  classes, complete prompt groups, or access to neighboring passages.
- **Reproducible feature provenance:** pinned checkpoint revisions, text
  checksums, persisted splits, dependency locking, and reusable OOF caches.
- **Offline deployment:** self-contained CPU and enhanced Kaggle notebooks
  that read the runtime test file and validate the resulting Parquet.
- **Executed verification:** seven automated tests passed; both notebooks
  reproduced their saved-model submission files exactly in local runs.

The enhanced notebook completed fresh extraction and retraining locally in
approximately **9.7 minutes**. The attribution stage processed **4,000 rows
in 15.1 seconds using precomputed encoder vectors**; that latter timing
excludes encoder extraction. Both measurements were made on an Apple M2 Pro
with 16 GB unified memory, not on Kaggle's scoring infrastructure.

## Project provenance

Project 2 was implemented from the supplied data and empirical validation,
without consulting research papers or reusing the first project's source,
trained models, feature caches, or split assignments. Off-the-shelf libraries
and the public pretrained ELECTRA checkpoint provide the underlying
numerical and neural-network infrastructure.

## What is implemented

- UTF-8 text and Parquet input/output validation.
- Character and word TF-IDF models.
- A separate representation of function words, word shapes, and punctuation.
- Explicit length, vocabulary, sentence, and Unicode punctuation features.
- Linear classifiers, boosted trees, and a randomized-tree comparison.
- Optional frozen `google/electra-large-discriminator` representations.
- Additional encoder classifiers with different regularization strengths.
- A training-time penalty that discourages dependence on prompt-level
  average features. This component needs prompt IDs only during training.
- An experimental classifier that fine-tunes the encoder's last four layers
  and a classification head, with an explicitly fixed training recipe.
- Small, documented ensemble searches on out-of-fold predictions.
- Reproducible final training, saved predictors, and offline Kaggle notebooks.

Predictions are made independently for each passage. There is no requirement
that a test batch contain all models, balanced labels, or other completions
from the same prompt. No class quotas or test-label estimates are imposed.

## Environment

Run from this directory:

```bash
uv sync --extra encoder
uv run python -m unittest -v test_pipeline
```

`uv.lock` records the resolved dependency versions. The classical pipeline
does not need PyTorch; `uv sync` installs its dependencies alone.
The encoder and fine-tuning commands support CUDA, Apple MPS, and CPU.
Fine-tuning on CPU is substantially slower.

The pretrained checkpoint is public and Apache-2.0 licensed. Its exact
revision is written to `checkpoints/electra-large/manifest.json`. Only model
files are downloaded. Competition text processing takes place locally.

## Reproduce development

Download and arrange the competition files as described in
[Dataset access and local file structure](#dataset-access-and-local-file-structure)
before running these commands.

```bash
uv run python workflow.py audit --output runs/audit
uv run python workflow.py cv --output runs/classical

uv run python encode.py --download
uv run python workflow.py cv --output runs/encoder --embeddings cache/electra.npz
```

Optional fine-tuning experiment, preferably on a CUDA machine with sufficient
memory:

```bash
uv run python neural.py --output runs/neural
uv run python workflow.py combine \
  --base runs/encoder --neural runs/neural --output runs/combined
```

In this session, two fine-tuning folds were completed. Their scores were
below the frozen encoder's scores on the same folds. The experiment was
stopped when the machine experienced memory pressure, and its incomplete
results were excluded from the final ensemble. The completed folds remain
available for inspection; they do not constitute five-fold CV.

The encoder cache verifies both text checksums and model configuration.
Training targets are never used to calculate frozen encoder vectors.
All vocabulary fitting, normalization, supervised fitting, and any
prompt-based regularization happen inside each training fold.

Fold predictions are cached. Re-running `cv` reuses existing compatible
components and evaluates newly added components. If changing an existing
component's behavior, increment `attribution.VERSION` and use a new run
directory; do not reuse predictions generated by a different implementation.

## Validation policy

The training data has 150 prompts. The fixed split reserves 30 prompts for
the holdout, and uses five folds across the other 120 prompts. All responses
to a prompt stay together. The seed is `71423`.

`cv` and `combine` select models using development predictions only.
`finalize` evaluates a fixed selection on the holdout and then refits that
selection on all training rows. It saves per-class recalls, a confusion
matrix, predictions, and an interval calculated by resampling entire
prompt groups.

Development scores are tuned scores, not independent estimates of final
performance. The holdout is excluded from this project's fitting and model
selection, although this dataset was used in earlier exploration and experiments.
Any reported Kaggle rank requires an actual Kaggle scoring run.

## Build deliverables

CPU fallback:

```bash
uv run python workflow.py finalize \
  --run runs/classical --output deliverables/cpu
uv run python make_notebook.py \
  --selection runs/classical/selection.json --output kaggle_cpu.ipynb
```

For the enhanced selection:

```bash
uv run python workflow.py finalize \
  --run runs/encoder --embeddings cache/electra.npz \
  --output deliverables/best
uv run python make_notebook.py \
  --selection runs/encoder/selection.json --output kaggle_submission.ipynb
```

If completing a future fine-tuning experiment and selecting it through the
`combine` command, use that new run directory instead.

The final results report records the actual run chosen and its measurements.
The model's selection metadata identifies its exact components and weights.

## Predict on another file locally

CPU model:

```bash
uv run python workflow.py predict \
  --model deliverables/cpu/predictor.joblib \
  --test Data/test.csv --output submission_cpu.parquet
```

Enhanced model:

```bash
uv run python encode.py --test path/to/test.csv --output cache/new_test.npz
uv run python workflow.py predict \
  --model deliverables/best/predictor.joblib \
  --test path/to/test.csv --embeddings cache/new_test.npz \
  --output submission.parquet
```

Keep the pretrained checkpoint available for a predictor containing a
fine-tuned component. Set `NTL_CHECKPOINT_DIR` to override its default path.
Load only model files you trust: joblib uses Python pickle internally.

## Run on Kaggle

1. Create a notebook for the competition and import the generated notebook.
2. Attach the competition data.
3. For the enhanced version, attach `checkpoints/electra-large/` as a
   private Kaggle dataset, including `manifest.json` and all model/tokenizer
   files. Select a GPU.
4. Disable internet and use **Save & Run All**.
5. Submit the notebook output named `submission.parquet`.

The CPU notebook needs no pretrained checkpoint. Both notebooks contain
their Python source and retrain the selected models, avoiding incompatibility
between a local model pickle and Kaggle's package versions.

`deliverables/kaggle_encoder_input.zip` contains the checkpoint folder ready
for creating that input dataset. Its extracted files must be visible to the
notebook; an unextracted ZIP alone is not sufficient.

The enhanced notebook computes training and test embeddings on the same
runtime device. It does not rely on feature rows for unseen test passages
already being cached. The checkpoint must be attached before the offline run.

The notebooks need NumPy, pandas, SciPy, scikit-learn, PyArrow, CatBoost,
joblib, and threadpoolctl; the enhanced one also needs PyTorch and
Transformers with ELECTRA support. Match the tested versions when possible.

## Files

| File | Purpose |
| --- | --- |
| `dataio.py` | Input contracts, checksums, grouped splits, Parquet output |
| `attribution.py` | Text/statistical features and ensemble components |
| `encode.py` | Frozen encoder feature extraction and checkpoint download |
| `neural.py` | Explicit fine-tuning experiment and neural inference |
| `workflow.py` | Audit, CV, selection, final training, prediction commands |
| `make_notebook.py` | Self-contained Kaggle notebook generator |
| `test_pipeline.py` | Leakage boundaries, batch independence, cache and output tests |
| `write_report.py` | Generate the results report from saved measurements |
| `runs/` | Split assignments, OOF predictions, selections, experiment logs |
| `deliverables/` | Trained predictors, holdout evaluations, sample submissions |

Competition data, feature caches, trained models, and text-bearing outputs
are excluded from version control by `.gitignore`.
