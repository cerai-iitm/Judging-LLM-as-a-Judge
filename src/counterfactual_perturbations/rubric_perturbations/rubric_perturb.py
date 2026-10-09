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
You are an expert editor for evaluation rubrics.

Your task is to generate a counterfactual version of a rubric criterion by reversing ONLY its evaluation criterion while preserving every other aspect of the rubric.

Rules:

1. Reverse the evaluation criterion so that the counterfactual represents the opposite judgment.
2. Make the fewest possible textual edits.
3. Preserve the topic, medical concepts, entities, examples, and context.
4. Preserve sentence structure whenever possible.
5. Preserve as much lexical overlap as possible.
6. Do NOT introduce new requirements.
7. Do NOT remove unrelated information.
8. Do NOT change the medical scenario or subject.
9. Do NOT make the writing unnatural or awkward.

The output should read like a natural human-written rubric whose intended judgment is the opposite of the original.

Output ONLY the rewritten rubric.
"""

USER_PROMPT = """
Original Rubric:

{rubric}

Generate a minimally edited counterfactual rubric that reverses the evaluation criterion while preserving all other content.
"""

def paraphrase_rubric(rubric):
    messages = [
        {"role": "system","content": SYSTEM_PROMPT,},
        {"role": "user","content": USER_PROMPT.format(rubric=rubric),},]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True,
    )

    return generated.strip()

healthbench = pd.read_csv("./perturb-rubrics-emnlp/healthbench.csv")
research_df = pd.read_csv("./perturb-rubrics-emnlp/research_df.csv")

research_df["paraphrased_rubric"] = pd.Series(dtype="string")
for i in range(1000):
    print("-"*50)
    print(f"Processing {i+1}/{len(research_df)}")
    input_text = ast.literal_eval(research_df.loc[i, "rubrics_parsed"])["criterion"]
    decoded_output = paraphrase_rubric(input_text)
    print("Input_text:", input_text)
    print("Generated text:", decoded_output)
    research_df.loc[i, "paraphrased_rubric"] = decoded_output
    if (i+1)%20 == 0:
        research_df.to_csv("./perturb-rubrics-emnlp/research_df_paraphrase.csv", index=False)
research_df.to_csv("./perturb-rubrics-emnlp/research_df_paraphrase.csv", index=False)

healthbench["paraphrased_rubric"] = pd.Series(dtype="string")
for i in range(1000):
    print("-"*50)
    print(f"Processing {i+1}/{len(healthbench)}")
    input_text = ast.literal_eval(healthbench.loc[i, "rubrics_parsed"])["criterion"]
    decoded_output = paraphrase_rubric(input_text)
    print("Input_text:", input_text)
    print("Generated text:", decoded_output)
    healthbench.loc[i, "paraphrased_rubric"] = decoded_output
    if (i+1)%20 == 0:
        healthbench.to_csv("./perturb-rubrics-emnlp/healthbench_paraphrase.csv", index=False)
healthbench.to_csv("./perturb-rubrics-emnlp/healthbench_paraphrase.csv", index=False)

# Now that rubric paraphrasing is done, run evaluation script as normal on perturbed rubrics.
