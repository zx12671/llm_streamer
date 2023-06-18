from transformers import AutoModelForCausalLM,AutoTokenizer

checkpoint = "C:/Users/xiangli/workspace/LaMini-GPT-1.5B/LaMini-GPT-124m"  # LaMini-GPT-124m
model = AutoModelForCausalLM.from_pretrained(f"{checkpoint}")
tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint}")
instruction = 'Please explain the question: \n"the effect of vitamin C"'

input_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

input_ids = tokenizer.encode(input_prompt, return_tensors="pt")

output = model.generate(input_ids, max_length=256, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
print("eos:"+str(output[0][-1]))
print("model generation:"+str(model.generation_config))