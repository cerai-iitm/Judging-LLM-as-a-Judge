import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import seaborn as sns
import time, ast
import re, json, random, math
from datetime import datetime, timedelta
from tqdm import tqdm
import os, pickle
from rapidfuzz import fuzz

# Hugging Face API token
from huggingface_hub import login
hf_token = "Configure your Hugging Face API token here"
login(token=hf_token)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

df = pd.read_csv("data/healthbench_probe/gemma_eval_judge.csv")
os.makedirs("results/judge", exist_ok=True)

system_prompt = """
You are an expert evaluator of medical rubric criteria.

Your task is to analyze a rubric criterion in isolation and predict
whether it is likely to be satisfied ("yes") or not satisfied ("no")
based purely on the semantic structure, framing, and implied expectations
of the rubric itself.

You are NOT given the generated answer or conversation context.

Instead, infer whether the rubric semantically represents:
- a commonly satisfied / positively aligned criterion ("yes")
or
- a commonly violated / negatively aligned criterion ("no")

Use only the wording, framing, semantic tone, and implied evaluative prior
present in the rubric criterion itself.

Respond with ONLY one word:

yes
or
no
"""

user_prompt = """
[Rubric Criterion]

{rubric}

Based only on this rubric criterion, predict whether it corresponds
to a satisfied evaluative condition.
"""

def get_messages(conversation):
    messages = []
    for turn in conversation:
        if turn["role"] == "user":
            role = "user"
        else:
            role = "assistant"

        messages.append({
            "role": role,
            "content": turn["content"]
        })
        
    return messages

def extract_answer(rubric, max_new_tokens = 3):
    complete_prompt = user_prompt.format(rubric=rubric)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": complete_prompt}
    ]    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    
    response = tokenizer.batch_decode(
        output[:, model_inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0].strip().lower()
    
    if "yes" in response:
        binary_label = 1
        prediction = "ENTAILED"

    elif "no" in response:
        binary_label = 0
        prediction = "CONTRADICTED"

    else:
        binary_label = None
        prediction = "UNKNOWN"

    return {
        "raw_response": response,
        "prediction": prediction,
        "no_context_binary_label": binary_label
    }
    
df["no_context_binary_label"] = pd.Series(dtype="object")
for i in tqdm(range(len(df))):
    print("-" * 80)
    print(f"Processing Conversation {i+1}/{len(df)}")
    print("-" * 80)
    rubric_text = df.loc[i, "rubric_text"]
    print("RUBRIC:", rubric_text)
    raw_output = extract_answer(rubric_text)
    print("RAW OUTPUT:", raw_output)
    df.loc[i, "no_context_binary_label"] = raw_output["no_context_binary_label"]
    if i % 50 == 0:
        df.to_csv("results/judge/llm_as_judge_gemma_r.csv", index=False)

df.to_csv("results/judge/llm_as_judge_gemma_r.csv", index=False)