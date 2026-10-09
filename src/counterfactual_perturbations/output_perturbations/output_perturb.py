import pandas as pd

import ast
import re
import os, ast
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Hugging Face API token
from huggingface_hub import login
hf_token = "Configure your Hugging Face API token here"
login(token=hf_token)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16).to(device)

SYSTEM_PROMPT = """
You are an expert assistant chatbot conversing with me.

You will be given a user request and a single rubric criterion.

Your task is to produce a high-quality response to the user request.

The response should be natural, coherent, factually accurate whenever possible, and helpful.

Do not mention the rubric or explicitly refer to the evaluation process.

Output only the response.
"""

USER_PROMPT_OBEY = """
User Request:

{conversation}

Rubric Criterion:

{rubric}

Generate a response that clearly satisfies the rubric criterion while remaining natural and helpful.
"""

USER_PROMPT_VIOLATE = """
User Request:

{conversation}

Rubric Criterion:

{rubric}

Generate a response that intentionally does NOT satisfy the rubric criterion.

The response should remain natural, coherent, and helpful whenever possible.

The failure should arise only because the rubric criterion is not satisfied.

Do not introduce unrelated mistakes or degrade the overall quality of the response.
"""

health_df = pd.read_csv("./rubric-perturb/healthbench_output_perturb.csv")
research_df = pd.read_csv("./rubric-perturb/research_output_perturb.csv")

def generate_response(conversation, rubric, mode="obey"):
    if mode == "obey":
        user_prompt = USER_PROMPT_OBEY
    elif mode == "violate":
        user_prompt = USER_PROMPT_VIOLATE
    else:
        raise ValueError("mode must be 'obey' or 'violate'")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT,},
        {"role": "user", "content": user_prompt.format(conversation=conversation,
                                                       rubric=rubric,)}]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.3, top_p=0.9,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    return generated.strip()

def get_messages(conversation):
    messages = []
    for turn in conversation:
        if turn["role"] == "user":
            role = "user"
        else:
            role = "assistant"
        messages.append({"role": role, "content": turn["content"]})
    return messages

health_df["obey_rubric"] = pd.Series(dtype=str)
health_df["violate_rubric"] = pd.Series(dtype=str)
for i in range(len(health_df)):
    print("-"*50)
    print(f"Processing {i+1}/{len(health_df)}")
    conversation = get_messages(ast.literal_eval(health_df.loc[i, "prompt"]))
    conv = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in conversation)

    rubric = health_df.loc[i, "selected_rubric"]
    obey_response = generate_response(conv, rubric, mode="obey")
    violate_response = generate_response(conv, rubric, mode="violate")
    print("OBEY RESPONSE:", obey_response)
    print("VIOLATE RESPONSE:", violate_response)
    health_df.loc[i, "obey_rubric"] = obey_response
    health_df.loc[i, "violate_rubric"] = violate_response

    if (i+1)%20 == 0:
        health_df.to_csv("./health_output_perturb.csv", index=False)
health_df.to_csv("./health_output_perturb.csv", index=False)

research_df["obey_rubric"] = pd.Series(dtype=str)
research_df["violate_rubric"] = pd.Series(dtype=str)
for i in range(len(research_df)):
    print("-"*50)
    print(f"Processing {i+1}/{len(research_df)}")
    conv = research_df.loc[i, "prompt"]
    rubric = research_df.loc[i, "selected_rubric"]
    obey_response = generate_response(conv, rubric, mode="obey")
    violate_response = generate_response(conv, rubric, mode="violate")
    print("OBEY RESPONSE:", obey_response)
    print("VIOLATE RESPONSE:", violate_response)
    research_df.loc[i, "obey_rubric"] = obey_response
    research_df.loc[i, "violate_rubric"] = violate_response

    if (i+1)%20 == 0:
        research_df.to_csv("./research_output_perturb.csv", index=False)
research_df.to_csv("./research_output_perturb.csv", index=False)