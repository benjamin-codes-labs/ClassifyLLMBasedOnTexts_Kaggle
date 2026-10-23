"""Original text attribution implementation, developed from dataset inspection.

Every prediction depends only on that passage and the fitted training model.
Changing test order, batch size, or other test passages cannot change it.
"""
from __future__ import annotations

import re
import time
from collections import Counter

import numpy as np
from catboost import CatBoostClassifier
from scipy import sparse
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import LinearSVC

from dataio import SEED

VERSION = "project2-1"
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
GLUE = tuple(sorted({
    "a", "an", "the", "and", "or", "but", "yet", "so", "as", "if", "than",
    "of", "to", "for", "in", "on", "at", "by", "with", "from", "into", "within",
    "without", "through", "between", "over", "under", "while", "where", "when",
    "which", "that", "this", "these", "those", "it", "its", "they", "their",
    "we", "our", "you", "your", "is", "are", "was", "were", "be", "been",
    "can", "could", "would", "may", "might", "will", "must", "not", "no",
    "only", "also", "even", "rather", "just", "more", "less", "each", "every",
    "both", "all", "because", "however", "therefore", "thus", "instead",
    "still", "often", "sometimes", "ultimately", "merely",
}))
PUNCTUATION = tuple(".,;:!?-'’\"“”()—–/…")


def grammatical_outline(text):
    def replace(match):
        word = match.group()
        if word.lower() in ENGLISH_STOP_WORDS:
            return word.lower()
        size = len(word)
        return ("CAP" if word[0].isupper() else "word") + str(min(size//3, 4))
    return WORD.sub(replace, text)


def numeric_features(texts):
    rows = []
    for text in texts:
        tokens = WORD.findall(text)
        n = max(1, len(tokens))
        size = max(1, len(text))
        vocabulary = Counter(t.lower() for t in tokens)
        lengths = np.array([len(t) for t in tokens] or [0])
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        sl = np.array([len(WORD.findall(s)) for s in sentences] or [0])
        frequency = np.array(list(vocabulary.values()) or [1])
        row = [
            np.log1p(size), np.log1p(n), size/n, len(vocabulary)/n,
            (frequency == 1).sum()/n, frequency.max()/n,
            (frequency*(frequency-1)).sum()/max(n*(n-1), 1),
            np.mean(lengths), np.std(lengths), np.median(lengths),
            np.quantile(lengths, .9), np.log1p(len(sentences)),
            np.mean(sl), np.std(sl), np.median(sl), sl.max(),
            sum(t[0].isupper() for t in tokens)/n,
            sum(t.isupper() and len(t)>1 for t in tokens)/n,
            sum(t.lower() in ENGLISH_STOP_WORDS for t in tokens)/n,
            sum(c.isdigit() for c in text)/size,
            sum(ord(c)>127 for c in text)/size,
        ]
        row.extend(np.mean(lengths == length) for length in range(1, 13))
        row.extend(vocabulary[word]/n for word in GLUE)
        for symbol in PUNCTUATION:
            count = text.count(symbol)
            row.extend([count/n, np.log1p(count)])
        # Separate punctuation habits near openings and endings.
        for segment in [text[:80], text[-80:]]:
            row.extend(segment.count(p)/max(1, len(segment)) for p in [",", ":", ";", "—", "’", "?"])
        # Sentence-level structural markers and repetition.
        beginnings = [WORD.findall(s.lower())[:1] for s in sentences]
        row.extend(sum(bool(b) and b[0] == w for b in beginnings)/max(1,len(sentences))
                   for w in ["the", "a", "it", "this", "yet", "but", "in", "as", "we"])
        row.extend([
            len(re.findall(r"\w—\w", text))/n,
            len(re.findall(r"\s—\s", text))/n,
            len(re.findall(r"\w-\w", text))/n,
            len(re.findall(r"\w['’]\w", text))/n,
            float(text.rstrip().endswith(".")),
            float(text.rstrip().endswith("?")),
            float(text.rstrip().endswith('"')),
        ])
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


class Representations:
    def fit(self, texts, vectors=None):
        self.char = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 6), min_df=3, max_features=130_000,
            lowercase=False, sublinear_tf=True, dtype=np.float32)
        self.word = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=50_000,
            token_pattern=r"(?u)\b[\w’']+\b", sublinear_tf=True, dtype=np.float32)
        self.outline = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 6), min_df=3, max_features=35_000,
            lowercase=False, sublinear_tf=True, dtype=np.float32)
        self.char.fit(texts)
        self.word.fit(texts)
        self.outline.fit([grammatical_outline(t) for t in texts])
        self.scaler = RobustScaler(quantile_range=(10, 90)).fit(numeric_features(texts))
        self.vector_scaler = None if vectors is None else StandardScaler().fit(vectors)
        return self

    def transform(self, texts, vectors=None):
        lexical = sparse.hstack([self.char.transform(texts), .65*self.word.transform(texts)], format="csr")
        numbers = numeric_features(texts)
        scaled = np.clip(self.scaler.transform(numbers), -12, 12).astype(np.float32)
        combined = sparse.hstack([
            lexical, .7*self.outline.transform([grammatical_outline(t) for t in texts]),
            sparse.csr_matrix(.06*scaled),
        ], format="csr")
        result = {"lexical": lexical, "combined": combined, "numbers": numbers, "scaled": scaled}
        if self.vector_scaler is not None:
            if vectors is None:
                raise ValueError("This model requires encoder vectors")
            result["vectors"] = self.vector_scaler.transform(vectors).astype(np.float32)
            result["vector_style"] = np.hstack([result["vectors"], scaled*2])
        return result


class PromptRegularizedRidge:
    """Discourage logits from tracking training prompt means.

    Extra rows represent the average completion for each training prompt.
    Their target is the same uniform class distribution. No prompt information
    or other test passages are needed when predicting.
    """
    requires_groups = True

    def __init__(self,alpha=1000,strength=5):
        self.alpha,self.strength=alpha,strength

    def fit(self,x,y,groups):
        if groups is None:
            raise ValueError("Training prompt IDs are required for this component")
        self.classes_=np.unique(y)
        target=np.eye(len(self.classes_))[y]-1/len(self.classes_)
        nuisance,targets=[],[]
        for group in np.unique(groups):
            rows=np.flatnonzero(groups==group)
            nuisance.append(x[rows].mean(0)*np.sqrt(len(rows)*self.strength))
            targets.append(np.zeros(len(self.classes_)))
        self.estimator=Ridge(alpha=self.alpha,fit_intercept=False,solver="lsqr")
        self.estimator.fit(np.vstack([x,nuisance]),np.vstack([target,targets]))
        return self

    def predict_proba(self,x):
        return softmax(self.estimator.predict(x)*8,axis=1)


def definitions(use_encoder):
    specs = {
        "lexical_svm": ("lexical", LinearSVC(C=1.0, dual=True, max_iter=5000,
                                            class_weight="balanced", random_state=SEED)),
        "structure_svm": ("combined", LinearSVC(C=1.0, dual=True, max_iter=5000,
                                               class_weight="balanced", random_state=SEED)),
        "structure_logistic": ("combined", LogisticRegression(C=4.0, max_iter=1500,
                                                              class_weight="balanced")),
        "style_boost": ("numbers", CatBoostClassifier(
            iterations=650, depth=5, learning_rate=.045, l2_leaf_reg=8,
            loss_function="MultiClass", random_seed=SEED, thread_count=4,
            verbose=False, allow_writing_files=False)),
        "style_forest": ("numbers", ExtraTreesClassifier(
            n_estimators=600, max_features=.6, min_samples_leaf=3,
            class_weight="balanced", n_jobs=4, random_state=SEED)),
    }
    if use_encoder:
        specs.update({
            "encoder_ridge": ("vectors", RidgeClassifier(alpha=100, solver="lsqr", class_weight="balanced")),
            "encoder_regularized": ("vector_style", RidgeClassifier(alpha=1000, solver="lsqr", class_weight="balanced")),
            "encoder_logistic": ("vector_style", LogisticRegression(C=.015, max_iter=1500,
                                                                    class_weight="balanced")),
            "encoder_logistic_strong": ("vector_style", LogisticRegression(C=.004, max_iter=1500,
                                                                           class_weight="balanced")),
            "encoder_logistic_light": ("vector_style", LogisticRegression(C=.06, max_iter=1500,
                                                                          class_weight="balanced")),
            "prompt_regularized": ("vector_style", PromptRegularizedRidge(alpha=1000,strength=5)),
            "prompt_regularized_light": ("vector_style", PromptRegularizedRidge(alpha=300,strength=5)),
        })
    return specs


class Predictor:
    def __init__(self, labels, selected=None):
        self.labels = list(labels)
        self.selected = selected

    def fit(self, texts, y, vectors=None, groups=None):
        self.features = Representations().fit(texts, vectors)
        matrices = self.features.transform(texts, vectors)
        self.members = {}
        self.neural_members = {}
        for name, (view, estimator) in definitions(vectors is not None).items():
            if self.selected is not None and name not in self.selected:
                continue
            started = time.perf_counter()
            if getattr(estimator,"requires_groups",False):
                estimator.fit(matrices[view], y, groups)
            else:
                estimator.fit(matrices[view], y)
            self.members[name] = (view, estimator)
            print(f"  {name}: {time.perf_counter()-started:.1f}s", flush=True)
        for name in self.selected or []:
            if name.startswith("neural_e"):
                from neural import NeuralModel
                self.neural_members[name] = NeuralModel(
                    len(self.labels), epochs=int(name.removeprefix("neural_e"))).fit(texts, y)
        return self

    def predict_members(self, texts, vectors=None):
        matrices = self.features.transform(texts, vectors)
        output = {}
        for name, (view, estimator) in self.members.items():
            if hasattr(estimator, "predict_proba"):
                p = estimator.predict_proba(matrices[view])
            else:
                score = estimator.decision_function(matrices[view])
                if score.ndim == 1:
                    score = np.column_stack([-score, score])
                # Fixed score temperatures; weights are selected using OOF data.
                temperature = 5.0 if "svm" in name else 4.0
                p = softmax(score*temperature, axis=1)
            aligned = np.zeros((len(texts), len(self.labels)), dtype=np.float64)
            aligned[:, np.asarray(estimator.classes_, dtype=int)] = p
            output[name] = aligned.astype(np.float32)
        for name, model in self.neural_members.items():
            output[name] = model.predict_proba(texts)
        return output


def weighted_predictions(members, weights):
    if any(v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Invalid ensemble weights")
    prediction = sum(members[name]*weight for name,weight in weights.items()) / sum(weights.values())
    return prediction/prediction.sum(axis=1, keepdims=True)
