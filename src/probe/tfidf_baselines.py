import os
import re
import json
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 250)

DATA_DIR = "data/healthbench_probe"
BINARY_ONLY = True
MIN_ROWS = 20
RANDOM_STATE = 42
NGRAM_RANGE = (1, 3)
MAX_FEATURES = 30000
OUTPUT_DIR = Path("results/tfidf_baselines")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_colname(c):
    return re.sub(r"[^a-z0-9]+", "", str(c).strip().lower())

def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x)
    x = x.replace("\n", " ")
    x = x.replace("\r", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x

def try_parse_label(series):
    s = series.copy()
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce")
    s = s.astype(str).str.strip().str.lower()
    mapping = {"true": 1, "false": 0, "yes": 1, "no": 0, "pass": 1, "fail": 0, 
               "positive": 1, "negative": 0, "met": 1, "not met": 0}
    mapped = s.map(mapping)
    numeric = pd.to_numeric(s, errors="coerce")
    return mapped.where(~mapped.isna(), numeric)

def infer_columns(df):
    cols = list(df.columns)
    norm_map = {c: normalize_colname(c) for c in cols}
    rubric_candidates = []
    label_candidates = []
    rubric_keywords = ["rubric", "rubrictext", "criteria", "criterion", "guideline", 
                       "instruction", "judgeprompt", "evalprompt"]

    label_keywords = ["label", "binarylabel", "score", "judgement", "judgment", "judgeoutput",
                      "decision", "verdict", "criteriamet", "result", "output"]
    for c in cols:
        nc = norm_map[c]
        if any(k in nc for k in rubric_keywords):
            rubric_candidates.append(c)
        if any(k in nc for k in label_keywords):
            label_candidates.append(c)

    if not rubric_candidates:
        text_like = []
        for c in cols:
            if df[c].dtype == "object":
                avg_len = df[c].astype(str).map(len).mean()
                text_like.append((c, avg_len))
        text_like = sorted(text_like, key=lambda x: -x[1])
        if text_like:
            rubric_candidates = [text_like[0][0]]

    if not label_candidates:
        scores = []
        for c in cols:
            vals = try_parse_label(df[c])
            nunique = vals.dropna().nunique()
            valid = vals.notna().mean()
            if 1 < nunique <= 10 and valid > 0.6:
                scores.append((c, nunique, valid))
        scores = sorted(scores, key=lambda x: (x[1], -x[2]))
        if scores:
            label_candidates = [scores[0][0]]
    rubric_col = rubric_candidates[0] if rubric_candidates else None
    label_col = label_candidates[0] if label_candidates else None
    return rubric_col, label_col

def robust_read_csv(path):
    attempts = [
        {"encoding": "utf-8", "engine": "python"},
        {"encoding": "latin-1", "engine": "python"},
        {"encoding": "cp1252", "engine": "python"}
    ]
    errors = []
    for cfg in attempts:
        try:
            df = pd.read_csv(path, encoding=cfg["encoding"], engine=cfg["engine"], on_bad_lines="skip")
            return df, None
        except Exception as e:
            errors.append(f"{cfg['encoding']} -> {str(e)}")
    return None, " | ".join(errors)

def parse_filename(fname):
    base = Path(fname).stem.lower()
    dataset_type = "unknown"
    if "hard" in base:
        dataset_type = "healthbench_hard"
    elif "eval" in base:
        dataset_type = "healthbench"
    source_model = base.split("_")[0]
    return source_model, dataset_type


models = {
    "dummy_majority": DummyClassifier(strategy="most_frequent"),
    "multinomial_nb": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES, min_df=2)),
        ("clf", MultinomialNB())
    ]),
    "logreg_l2": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES, min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
}

all_results = []
file_summaries = []
all_errors = []
paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
print("Found files:", len(paths))

for p in paths:
    fname = os.path.basename(p)
    print("PROCESSING:", fname)
    source_model, dataset_type = parse_filename(fname)
    df, err = robust_read_csv(p)
    if df is None:
        print("FAILED TO LOAD")
        file_summaries.append({"file": fname, "loaded": False, "reason": err})
        continue
    print("Loaded shape:", df.shape)
    if len(df) < MIN_ROWS:
        print("Too few rows")
        file_summaries.append({"file": fname, "loaded": False, "reason": f"too few rows: {len(df)}"})
        continue

    rubric_col, label_col = infer_columns(df)
    print("Rubric column:", rubric_col)
    print("Label column:", label_col)
    if rubric_col is None or label_col is None:
        print("Could not infer columns")
        file_summaries.append({"file": fname, "loaded": False, "reason": "column inference failed"})
        continue

    tmp = df[[rubric_col, label_col]].copy()
    tmp.columns = ["rubric_text", "judge_label_raw"]
    tmp["judge_label"] = try_parse_label(tmp["judge_label_raw"])
    tmp["rubric_text"] = tmp["rubric_text"].map(clean_text)
    tmp = tmp.dropna(subset=["rubric_text", "judge_label"])
    tmp = tmp[tmp["rubric_text"].str.len() > 0]

    if BINARY_ONLY:
        tmp = tmp[tmp["judge_label"].isin([0, 1])]
    print("Usable rows:", len(tmp))
    if len(tmp) < MIN_ROWS:
        print("Too few usable rows")
        continue

    X = tmp["rubric_text"]
    y = tmp["judge_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    for clf_name, model in models.items():
        print("Running:", clf_name)
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            row = {
                "dataset": dataset_type,
                "source_model": source_model,
                "classifier": clf_name,
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro"),
                "weighted_f1": f1_score(y_test, pred, average="weighted"),
                "rows": len(tmp)
            }
            if (len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba")):
                prob = model.predict_proba(X_test)[:, 1]
                row["roc_auc"] = roc_auc_score(y_test, prob)
            all_results.append(row)

            err_df = pd.DataFrame({
                "rubric_text": X_test.values,
                "gold": y_test.values,
                "pred": pred
            })

            err_df["correct"] = (err_df["gold"] == err_df["pred"]).astype(int)
            errors = err_df[err_df["correct"] == 0]
            err_out = OUTPUT_DIR / (f"errors_{source_model}_{dataset_type}_{clf_name}.csv")
            errors.to_csv(err_out, index=False)

        except Exception as e:
            print("FAILED:", str(e))
            all_errors.append({"file": fname, "classifier": clf_name, "error": str(e)})

    file_summaries.append({
        "file": fname,
        "loaded": True,
        "dataset": dataset_type,
        "source_model": source_model,
        "rows": len(tmp),
        "rubric_col": rubric_col,
        "label_col": label_col
    })

results_df = pd.DataFrame(all_results)
summary_df = pd.DataFrame(file_summaries)
errors_df = pd.DataFrame(all_errors)
results_df.to_csv(OUTPUT_DIR / "per_model_results.csv", index=False)
summary_df.to_csv(OUTPUT_DIR / "file_summary.csv", index=False)
errors_df.to_csv(OUTPUT_DIR / "pipeline_errors.csv", index=False)

print("\nResults shape:")
print(results_df.shape)
print("\nResults head:")
print(results_df.head())
print("\nSaved files:")
print(list(OUTPUT_DIR.glob("*")))


# Plotting and understanding features
ANALYSIS_DIR = Path("results/tfidf_baselines")
OUT_DIR = Path("figures/tfidf_baselines")
OUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")
POS = "#2ca02c"
NEG = "#d62728"
BLUE = "#1f77b4"

def save_png(name):
    plt.tight_layout()
    out_path = OUT_DIR / f"{name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("saved:", out_path)
    plt.close()

# LOAD CLASSIFIER RESULTS
results_path = ANALYSIS_DIR / "per_model_results.csv"
if not results_path.exists():
    raise ValueError(f"Missing file: {results_path}")

results_df = pd.read_csv(results_path, engine="python", encoding="utf-8", on_bad_lines="skip")
print("\nLoaded classifier results shape:", results_df.shape)

# Create a consistent model_name column for plotting
if "model_name" not in results_df.columns:
    if {"source_model", "classifier"}.issubset(results_df.columns):
        results_df["model_name"] = (
            results_df["source_model"].astype(str)
            + "_"
            + results_df["classifier"].astype(str)
        )
    elif "classifier" in results_df.columns:
        results_df["model_name"] = results_df["classifier"].astype(str)
    else:
        results_df["model_name"] = results_df.index.astype(str)

# FIGURE 1 — MACRO F1
if "macro_f1" in results_df.columns:
    plot_df = results_df.sort_values("macro_f1", ascending=True)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=plot_df, y="model_name", x="macro_f1", color=BLUE)
    plt.xlim(0, 1)
    plt.title("Macro F1 across models")
    plt.xlabel("Macro F1")
    plt.ylabel("")
    save_png("macro_f1_across_models")
else:
    print("macro_f1 column not found; skipping macro f1 plot")

# FIGURE 2 — ACCURACY
if "accuracy" in results_df.columns:
    plot_df = results_df.sort_values("accuracy", ascending=True)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=plot_df, y="model_name", x="accuracy", color=POS)
    plt.xlim(0, 1)
    plt.title("Accuracy across models")
    plt.xlabel("Accuracy")
    plt.ylabel("")
    save_png("accuracy_across_models")
else:
    print("accuracy column not found; skipping accuracy plot")

# FIGURE 3 — ROC AUC
if "roc_auc" in results_df.columns:
    plot_df = results_df.sort_values("roc_auc", ascending=True)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=plot_df, y="model_name", x="roc_auc", color=NEG)
    plt.xlim(0, 1)
    plt.title("ROC AUC across models")
    plt.xlabel("ROC AUC")
    plt.ylabel("")
    save_png("rocauc_across_models")
else:
    print("roc_auc not present for any model; skipping roc auc plot")

# FIGURE 4 — ALL METRICS TOGETHER
metric_cols = [
    c for c in ["accuracy", "macro_f1", "weighted_f1", "roc_auc"]
    if c in results_df.columns
]

if metric_cols:
    long_df = results_df.melt(
        id_vars="model_name",
        value_vars=metric_cols,
        var_name="metric",
        value_name="score"
    )
    long_df = long_df.reset_index(drop=True)

    plt.figure(figsize=(14, 8))
    sns.barplot(data=long_df, x="metric", y="score", hue="model_name")
    plt.ylim(0, 1)
    plt.title("Classifier performance summary")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    save_png("classifier_performance_summary")
else:
    print("No metric columns available to plot combined summary")

# PROBE / FEATURE FIGURES
def clean_text_func(x):
    if pd.isna(x):
        return ""
    x = str(x)
    x = x.replace("\n", " ").replace("\r", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x

def add_text_features(df):
    out = df.copy()
    out["char_len"] = out["rubric_text"].str.len()
    out["word_len"] = out["rubric_text"].str.split().map(len)
    out["digit_count"] = out["rubric_text"].str.count(r"\d")
    out["question_count"] = out["rubric_text"].str.count(r"\?")
    out["justification_count"] = (out["rubric_text"].str.lower().str.count(r"justification"))
    out["step_count"] = (out["rubric_text"].str.lower().str.count(r"\bstep\b"))
    out["contains_class"] = (out["rubric_text"].str.lower().str.contains(r"\bclass\b").astype(int))
    out["contains_fail"] = (out["rubric_text"].str.lower().str.contains(r"\bfail\b").astype(int))
    out["contains_pass"] = (out["rubric_text"].str.lower().str.contains(r"\bpass\b").astype(int))
    out["contains_should"] = (out["rubric_text"].str.lower().str.contains(r"\bshould\b").astype(int))
    out["contains_not"] = (out["rubric_text"].str.lower().str.contains(r"\bnot\b").astype(int))
    return out

all_rows = []
csv_paths = sorted(ANALYSIS_DIR.glob("*.csv"))

for p in csv_paths:
    if p.name == "per_model_results.csv":
        continue
    try:
        df = pd.read_csv( p, engine="python", encoding="latin-1", on_bad_lines="skip")
    except Exception as e:
        print("failed to read:", p.name, e)
        continue
    if df.shape[1] < 2:
        continue

    text_col, label_col = df.columns[:2]
    tmp = df[[text_col, label_col]].copy()
    tmp.columns = ["rubric_text", "judge_label"]
    tmp["rubric_text"] = tmp["rubric_text"].astype(str).map(clean_text_func)
    tmp["judge_label"] = pd.to_numeric(tmp["judge_label"], errors="coerce")
    tmp = tmp.dropna()
    tmp = tmp[tmp["judge_label"].isin([0, 1])]

    if len(tmp) < 20:
        continue

    tmp = add_text_features(tmp)
    tmp["source_file"] = p.name
    all_rows.append(tmp)

if not all_rows:
    print("No probe rows loaded; skipping probe plots")
    sys.exit(0)

probe_df = pd.concat(all_rows, ignore_index=True)
print("\nLoaded probe rows:", len(probe_df))

probe_cols = [
    "char_len",
    "word_len",
    "digit_count",
    "question_count",
    "justification_count",
    "step_count",
    "contains_class",
    "contains_fail",
    "contains_pass",
    "contains_should",
    "contains_not"
]

probe_table = []

for col in probe_cols:
    grouped = probe_df.groupby("judge_label")[col].mean().to_dict()
    probe_table.append({
        "feature": col,
        "mean_label_0": grouped.get(0, np.nan),
        "mean_label_1": grouped.get(1, np.nan),
        "difference_1_minus_0": (
            grouped.get(1, np.nan) - grouped.get(0, np.nan)
        )
    })

probe_results = pd.DataFrame(probe_table)

# FIGURE 5 — PROBE DIFFERENCES
plot_df = probe_results.sort_values("difference_1_minus_0")
colors = [
    NEG if x < 0 else POS
    for x in plot_df["difference_1_minus_0"]
]

plt.figure(figsize=(12, 8))
sns.barplot(
    data=plot_df,
    y="feature",
    x="difference_1_minus_0",
    palette=colors
)
plt.axvline(0, color="black", linestyle="--")
plt.title("Shortcut probe features\n(label 1 - label 0)")
plt.xlabel("Difference in mean feature value")
plt.ylabel("")
save_png("shortcut_probe_differences")

# FIGURE 6 — NUMERIC FEATURE DISTRIBUTIONS
numeric_features = [
    "char_len",
    "word_len",
    "digit_count",
    "question_count"
]

for feat in numeric_features:
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=probe_df, x="judge_label", y=feat)
    plt.title(f"{feat} distribution by label")
    plt.xlabel("Label")
    plt.ylabel(feat)
    save_png(f"{feat}_distribution")

# FIGURE 7 — BINARY FEATURE FREQUENCIES
binary_features = [
    "contains_class",
    "contains_fail",
    "contains_pass",
    "contains_should",
    "contains_not"
]

binary_rows = []

for feat in binary_features:
    vals = probe_df.groupby("judge_label")[feat].mean().reset_index()
    vals["feature"] = feat
    vals = vals.rename(columns={feat: "value"})
    binary_rows.append(vals)

binary_plot = pd.concat(binary_rows, ignore_index=True)

plt.figure(figsize=(14, 8))
sns.barplot(data=binary_plot, x="feature", y="value", hue="judge_label")
plt.title("Binary lexical cue frequencies")
plt.xlabel("")
plt.ylabel("Mean frequency")
save_png("binary_probe_features")

print("\nFinished generating all classifier + probe figures.")