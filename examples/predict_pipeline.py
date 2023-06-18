from transformers import AutoModelForCausalLM, AutoTokenizer,TextStreamer
from transformers import pipeline
import streamlit as st


def get_model():
    # checkpoint = "LaMini-GPT-1.5B"  #LaMini-GPT-124m
    checkpoint = "C:/Users/xiangli/workspace/LaMini-GPT-1.5B/LaMini-GPT-124m"  # LaMini-GPT-124m

    tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint}")

    model = AutoModelForCausalLM.from_pretrained(f"{checkpoint}")
    model = model.eval()
    return tokenizer, model

tokenizer, model = get_model()
model = pipeline('text-generation', model = model, tokenizer=tokenizer)

instruction = 'Please explain the question: \n"the effect of vitamin C"'

input_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

generated_text = model(input_prompt, max_length=64, do_sample=True)[0]['generated_text']
print("Response", generated_text)