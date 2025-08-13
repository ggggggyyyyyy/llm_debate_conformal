from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(r"models\Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained(r"models\Llama-2-7b-chat-hf")
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=5)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))