import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as mpatches
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
import umap.umap_ as umap
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


SAVE_PATH = "checkpoints/pubmedbert_gemma_eval"
tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
model = AutoModelForSequenceClassification.from_pretrained(SAVE_PATH)

model.eval()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

df = pd.read_csv("data/healthbench_probe/gemma_eval_judge.csv")
df = df.dropna(subset=["rubric_text", "binary_label"]).reset_index(drop=True)
df["rubric_text"] = df["rubric_text"].astype(str)
df["binary_label"] = df["binary_label"].astype(int)
print("\nClass Distribution:")
print(df["binary_label"].value_counts(normalize=True))
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["rubric_text"].tolist(),
    df["binary_label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["binary_label"]
    )

df2 = pd.read_csv("data/healthbench_probe/gemma_hard_judge.csv")
df2 = df2.dropna(subset=["rubric_text", "binary_label"]).reset_index(drop=True)
df2["rubric_text"] = df2["rubric_text"].astype(str)
df2["binary_label"] = df2["binary_label"].astype(int)
print("\nClass Distribution:")
print(df2["binary_label"].value_counts(normalize=True))
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df2["rubric_text"].tolist(),
    df2["binary_label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df2["binary_label"]
    )

model.eval()
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(
        token_embeddings * input_mask_expanded,
        1)/ torch.clamp(
            input_mask_expanded.sum(1),
            min=1e-9
            )
    
def get_embeddings(texts, batch_size=16):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():

            outputs = model.bert(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"]
            )
            embeddings = mean_pooling(
                outputs,
                encoded["attention_mask"]
            )

        all_embeddings.append(embeddings.cpu().numpy()),
    return np.vstack(all_embeddings)

embeddings1 = get_embeddings(df["rubric_text"].tolist())
labels1 = df["binary_label"].values

embeddings2 = get_embeddings(df2["rubric_text"].tolist())
labels2 = df2["binary_label"].values

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

embedding_2d_1 = reducer.fit_transform(embeddings1)
embedding_2d_2 = reducer.fit_transform(embeddings2)

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CLASS_COLORS = {0: "#2166AC", 1: "#D6604D"}
CLASS_LABELS = {0: "Class 0",  1: "Class 1"}
ALPHA        = 0.55
POINT_SIZE   = 18
EDGE_WIDTH   = 0.25
EDGE_COLOR   = "white"

def _class_colors(labels):
    return [CLASS_COLORS[int(l)] for l in labels]

def _legend_handles():
    return [
        mpatches.Patch(facecolor=CLASS_COLORS[0], edgecolor="#444", linewidth=0.6,
                       label=CLASS_LABELS[0]),
        mpatches.Patch(facecolor=CLASS_COLORS[1], edgecolor="#444", linewidth=0.6,
                       label=CLASS_LABELS[1]),
    ]

def plot_umap(embedding_2d, labels, title: str, ax: plt.Axes) -> None:
    colors = _class_colors(labels)

    ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=colors,
        s=POINT_SIZE,
        alpha=ALPHA,
        linewidths=EDGE_WIDTH,
        edgecolors=EDGE_COLOR,
        rasterized=True,
    )

    ax.set_xlabel("UMAP Dimension 1", labelpad=6)
    ax.set_ylabel("UMAP Dimension 2", labelpad=6)
    ax.set_title(title, pad=10, fontweight="bold")

    ax.grid(True, linestyle="--", linewidth=0.4, color="#cccccc", alpha=0.7)
    ax.set_axisbelow(True)

    ax.legend(
        handles=_legend_handles(),
        title="Label",
        title_fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        loc="upper right",
    )
    ax.set_facecolor("#F7F7F7")


fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True,)

plot_umap(embedding_2d_1, labels1, title="UMAP Projection for HealthBench Eval Probe", ax=axes[0])
plot_umap(embedding_2d_2, labels2, title="UMAP Projection for HealthBench Hard Probe", ax=axes[1])
plt.show()