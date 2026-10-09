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
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,).to(device)

df = pd.read_csv("data/responses/gemma_eval_responses.csv")
os.makedirs("results/judge", exist_ok=True)

system_prompt = """
You are an expert clinical evaluator.

Your task is to determine whether a generated medical answer satisfies a rubric criterion.

Carefully compare:
1. The rubric criterion
2. The generated answer

Use semantic reasoning and clinical understanding.

A rubric is ENTAILED if:
- the answer clearly satisfies the criterion,
- or semantically fulfills the requested requirement.

A rubric is CONTRADICTED if:
- the answer fails to satisfy the criterion,
- omits essential requested information,
- or gives conflicting advice.

Respond in a single word as to whether the answer generated satisfies the rubric or not. Reply with ONLY:

yes
or
no
"""

user_prompt = """
[Conversation]

{conversation}

[Generated Answer]

{answer}

[Rubric Criterion]

{rubric}

Does the answer satisfy the rubric criterion?
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

def extract_answer(conversation, answer, rubric, max_new_tokens = 3):
    complete_prompt = user_prompt.format(conversation=conversation, answer=answer, rubric=rubric)
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
        "binary_label": binary_label
    } 
    
df["judge_outputs"] = pd.Series(dtype="object")
for i in tqdm(range(len(df))):
    print("-" * 80)
    print(f"Processing Conversation {i+1}/{len(df)}")
    print("-" * 80)
    conversation = ast.literal_eval(df.loc[i, "prompt"])
    conversation_text = ""
    
    for turn in conversation:
        role = turn["role"].upper()
        content = turn["content"]
        conversation_text += f"{role}: {content}\n\n"

    generated_answer = df.loc[i, "generated_answer"]
    rubrics = df.loc[i, "rubrics"]

    if isinstance(rubrics, str):
        rubrics = ast.literal_eval(rubrics)
    sample_results = []

    for rubric_id, rubric_obj in enumerate(rubrics):
        rubric_text = rubric_obj["criterion"]
        print(f"\nRubric {rubric_id+1}")
        print("RUBRIC:", rubric_text)
        raw_output = extract_answer(conversation_text, generated_answer, rubric_text)
        print("RAW OUTPUT:", raw_output)

        result = {
            "rubric_id": rubric_id,
            "rubric_text": rubric_text,
            "rubric_tags": rubric_obj.get("tags", []),
            "raw_judge_output": raw_output,
            "prediction": raw_output["prediction"],
            "binary_label": raw_output["binary_label"]
        }

        sample_results.append(result)

    df.at[i, "judge_outputs"] = json.dumps(sample_results)
    if i % 50 == 0:
        df.to_csv("results/judge/llm_as_judge_gemma_rr.csv", index=False)

df.to_csv("results/judge/llm_as_judge_gemma_rr.csv", index=False)