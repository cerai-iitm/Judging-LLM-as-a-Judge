# Judging LLM-as-a-Judge
### Concerning Rubric Artifacts in LLM-based Automated Text Generation Evaluation

**Anshul Bagaria, Sowmya S Sundaram, Gokul S Krishnan, Balaraman Ravindran**
Centre for Responsible AI (CeRAI), Wadhwani School of Data Science and AI, IIT Madras
*EMNLP 2026*

---

LLM-as-a-Judge pipelines assume that verdicts come from reasoning over a candidate response with respect to a rubric. We test this assumption with a **rubric-only probe**: a classifier that sees *only* the rubric text and predicts the judge's verdict, with no access to the conversation or the evaluated response.

```
rubric text r  ──►  probe classifier  ──►  judge label y        (conversation and candidate response withheld)
```

If the rubric only guided judgment, `p(y | r)` would be at chance (0.5). Instead:

| Experiment | Result |
|---|---|
| Rubric-only probe, HealthBench (PubMedBERT) | Balanced accuracy **0.80–0.88** |
| Rubric-only probe, ResearchRubrics | Signal persists outside the medical domain |
| Output perturbation: response reversed, rubric fixed | Judge flips as expected in only **37.7%** of pairs |
| Rubric perturbation: criterion reversed, response fixed | Judge flips as expected in only **16.8%** (Gemma) / **32.2%** (LLaMA) of pairs |

## Setup

```bash
git clone <repo-url>
cd JudgingLLM-as-a-Judge
pip install -r requirements.txt
```

- Run every script **from the repository root**, e.g. `python src/probe/probe_weighted_eval.py`. Scripts read from `data/` and write to `results/`, `figures/` and `checkpoints/`, which are created as needed.
- Scripts that download gated models call `huggingface_hub.login`. Replace the `hf_token` placeholder at the top of the script with your own [Hugging Face token](https://huggingface.co/settings/tokens). Do not commit it.
- Original experiments: NVIDIA A100 GPUs, Python 3.10, `transformers` 4.38.2, `torch` 2.1.

| Role | Model |
|---|---|
| Candidate responses | `google/gemma-7b-it`, `meta-llama/Llama-3.1-8B-Instruct`, `google/medgemma-1.5-4b-it`, `MMed-Llama-3-8B` |
| Judge | `Qwen/Qwen2.5-7B-Instruct` |
| Rubric-only probe | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` |
| Counterfactual generation | `mistralai/Mistral-7B-Instruct-v0.3` |

## Repository structure

```
JudgingLLM-as-a-Judge/
├── data/                                  # column descriptions in data/README.md
│   ├── healthbench_probe/                 # rubric-only probe datasets, {model}_{eval,hard}_judge.csv
│   └── researchrubrics_probe/             # ResearchRubrics prompts, rubrics and candidate answers
├── src/
│   ├── judge/                             # 1. LLM-as-a-Judge labels
│   ├── probe/                             # 2. rubric-only probe classifiers
│   ├── counterfactual_perturbations/      # 3. counterfactual experiments (scripts + data)
│   │   ├── output_perturbations/
│   │   └── rubric_perturbations/
│   └── analysis/                          # 4. semantic analysis of rubrics
├── requirements.txt
└── README.md
```

## Pipeline

### 1. Judge labels: `src/judge/`

| Script | Description |
|---|---|
| `judge_response_and_rubric.py` | Qwen judge scores each candidate response against each rubric criterion, given the conversation. Produces `binary_label`. |
| `judge_rubric_only.py` | Qwen judge sees only the rubric text. Produces `no_context_binary_label`. |

### 2. Rubric-only probe: `src/probe/` (§4, App. B–D)

| Script | Paper | Description |
|---|---|---|
| `probe_weighted_eval.py`, `probe_weighted_hard.py` | Fig. 3 (WC), Table 2 | PubMedBERT probe with class-weighted loss, 5-fold CV |
| `probe_balanced_eval.py`, `probe_balanced_hard.py` | Fig. 3 (BS) | PubMedBERT probe on balanced subsamples |
| `cross_dataset.py` | Fig. 9 | Train on HB-Eval and test on HB-Hard, and the reverse |
| `tfidf_baselines.py` | Fig. 8, 11 | TF-IDF features with logistic regression, naïve Bayes and a majority-class baseline |

The probe scripts are configured for the Gemma cohort. To run another cohort, change the CSV prefix at the top of the script to `llama_`, `medgemma_` or `medllama_`.

### 3. Counterfactual perturbations: `src/counterfactual_perturbations/` (§5.1–5.2, App. E)

**Output perturbation.** The conversation and rubric stay fixed while the response is reversed.

| File | Description |
|---|---|
| `output_perturbations/output_perturb.py` | Mistral-7B writes one response that **satisfies** the rubric and one that **violates** it. Both are then scored by the judge. |
| `output_perturbations/healthbench_output_perturb.csv` | 500 sampled HealthBench conversations, one `selected_rubric` each |
| `output_perturbations/research_output_perturb.csv` | 101 ResearchRubrics prompts, one `selected_rubric` each |

**Rubric perturbation.** The conversation and response stay fixed while the rubric criterion is reversed.

| File | Description |
|---|---|
| `rubric_perturbations/rubric_perturb.py` | Mistral-7B rewrites each rubric with a minimal edit that **reverses** its criterion (greedy decoding) |
| `rubric_perturbations/health_rubric_perturb.csv` | 1,000 HealthBench rubrics with the original (`rubric_text`), the reversed rubric (`paraphrased_rubric`) and judge verdicts for both (`{gemma,llama}_rubric_label`, `{gemma,llama}_para_rubric_label`) |
| `rubric_perturbations/research_rubric_perturb.csv` | The same for 1,000 ResearchRubrics criteria |

A judge that tracks the criterion should flip its verdict in every pair, i.e. `*_rubric_label ≠ *_para_rubric_label`.

### 4. Rubric semantics: `src/analysis/` (§5.3, App. F)

| Script | Paper | Description |
|---|---|---|
| `umap_embeddings.py` | Fig. 6 | UMAP of probe embeddings, HB-Eval vs. HB-Hard |
| `topic_modeling.py` | Fig. 13 | BERTopic fitted separately for each judge label |
| `integrated_gradients.py` | App. F | Token attributions for the probe classifier |

These scripts load a trained probe from `checkpoints/pubmedbert_gemma_eval`.

## Data

- **HealthBench** (Arora et al., 2025): 5,000 healthcare conversations (1,000 in the Hard subset), each with physician-written, conversation-specific rubrics. MIT licence. The raw `.jsonl` files are not included here; download them from OpenAI's [simple-evals](https://github.com/openai/simple-evals) repository.
- **ResearchRubrics** (Sharma et al., 2026): 101 research prompts with about 2.6k instance-specific rubric criteria.

See [`data/README.md`](data/README.md) for file and column descriptions.

<!-- ## Citation

```bibtex
@inproceedings{bagaria2026judging,
  title     = {Judging {LLM}-as-a-Judge: Concerning Rubric Artifacts in {LLM}-based Automated Text Generation Evaluation},
  author    = {Bagaria, Anshul and Sundaram, Sowmya S and Krishnan, Gokul S and Ravindran, Balaraman},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
``` -->

## Licence and usage

HealthBench is released under the MIT licence. Model weights are used under their respective licences: Llama 3.1 Community License, Gemma Terms of Use, Health AI Developer Foundations Terms of Use, and CC-BY-NC-ND for MMed-Llama-3. See Appendix A.1 of the paper.
