# Subsampling Approach for Balanced Evaluation of Classification

import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

df = pd.read_csv("data/healthbench_probe/gemma_hard_judge.csv")
df = df.dropna(subset=["rubric_text", "binary_label"]).reset_index(drop=True)
df["rubric_text"] = df["rubric_text"].astype(str)
df["binary_label"] = df["binary_label"].astype(int)
print("\nOriginal Class Distribution:")
print(df["binary_label"].value_counts())

df_majority = df[df["binary_label"] == 0].reset_index(drop=True)
df_minority = df[df["binary_label"] == 1].reset_index(drop=True)
minority_size = len(df_minority)
print("\nMinority Size:", minority_size)

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

RESULTS_DIR = "results/probe/gemma"
os.makedirs(RESULTS_DIR, exist_ok=True)
METRIC_COLS = ["accuracy", "f1", "precision", "recall", "balanced_accuracy"]

class ClinicalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "precision": float(precision_score(labels, preds)),
        "recall": float(recall_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds))
    }
    
def write_fold_results(f, run, fold, results, report, cm):
    """Write a single run/fold block into an already-open file handle."""
    f.write("=" * 80 + "\n")
    f.write(f"RUN {run} | FOLD {fold}\n")
    f.write("=" * 80 + "\n\n")
    f.write("METRICS\n\n")
    for k, v in results.items():
        f.write(f"  {k}: {v}\n")
    f.write("\n")
    f.write("CLASSIFICATION REPORT\n\n")
    f.write(report)
    f.write("\n\n")
    f.write("CONFUSION MATRIX\n\n")
    f.write(str(cm))
    f.write("\n\n")
    
def write_summary(f, results_df):
    """Write the aggregate summary block into an already-open file handle."""
    f.write("=" * 80 + "\n")
    f.write("ALL FOLD RESULTS\n")
    f.write("=" * 80 + "\n\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")

    for label, agg_fn in [("MEAN", results_df[METRIC_COLS].mean()),
                           ("STD",  results_df[METRIC_COLS].std())]:
        f.write("=" * 80 + "\n")
        f.write(f"{label} METRICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(agg_fn))
        f.write("\n\n")

N_RUNS = 1
all_results = []
output_path = os.path.join(RESULTS_DIR, "balanced_hard_rubrics.txt")

with open(output_path, "w") as out_f:
    out_f.write("=" * 80 + "\n")
    out_f.write("BALANCED CROSS VALIDATION — PUBMEDBERT\n")
    out_f.write("=" * 80 + "\n\n")

    for run in range(N_RUNS):
        print("\n" + "=" * 70)
        print(f"BALANCED RUN {run + 1}")
        print("=" * 70)
        sampled_majority = df_majority.sample(n=minority_size, random_state=SEED + run)
        balanced_df = pd.concat(
            [sampled_majority, df_minority],
            axis=0
        ).sample(
            frac=1,
            random_state=SEED + run
        ).reset_index(drop=True)

        print("\nBalanced Distribution:")
        print(balanced_df["binary_label"].value_counts())
        texts = balanced_df["rubric_text"].tolist()
        labels = balanced_df["binary_label"].tolist()
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
            print("\n" + "-" * 50)
            print(f"Run {run + 1} | Fold {fold}")
            print("-" * 50)

            train_texts  = [texts[i] for i in train_idx]
            val_texts    = [texts[i] for i in val_idx]
            train_labels = [labels[i] for i in train_idx]
            val_labels   = [labels[i] for i in val_idx]

            train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
            val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)
            train_dataset = ClinicalDataset(train_encodings, train_labels)
            val_dataset = ClinicalDataset(val_encodings, val_labels)

            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

            training_args = TrainingArguments(
                output_dir=f"checkpoints/pubmedbert_gemma_hard_run_{run+1}_fold_{fold+1}",
                eval_strategy="epoch",
                save_strategy="no",
                learning_rate=2e-5,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=6,
                weight_decay=0.01,
                logging_steps=20,
                fp16=True,
                report_to="none",
                dataloader_pin_memory=False
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics
            )

            trainer.train()
            results = trainer.evaluate()
            predictions_output = trainer.predict(val_dataset)
            preds = np.argmax(predictions_output.predictions, axis=-1)
            labels_fold = predictions_output.label_ids

            report = classification_report(labels_fold, preds)
            cm = confusion_matrix(labels_fold, preds)
            print("\nClassification Report:\n", report)
            print("\nConfusion Matrix:\n",      cm)
            
            write_fold_results(out_f, run + 1, fold, results, report, cm)
            out_f.flush()

            fold_result = {
                "run": run + 1,
                "fold": fold,
                "accuracy": results["eval_accuracy"],
                "f1": results["eval_f1"],
                "precision": results["eval_precision"],
                "recall": results["eval_recall"],
                "balanced_accuracy": results["eval_balanced_accuracy"]
            }
            all_results.append(fold_result)

    results_df = pd.DataFrame(all_results)
    write_summary(out_f, results_df)

print("\n" + "=" * 70)
print("FINAL BALANCED CROSS VALIDATION RESULTS")
print("=" * 70)
print(results_df.to_string(index=False))
print("\nAVERAGE METRICS:\n", results_df[METRIC_COLS].mean())
print("\nSTANDARD DEVIATIONS:\n", results_df[METRIC_COLS].std())
print(f"\nSaved results to: {output_path}")