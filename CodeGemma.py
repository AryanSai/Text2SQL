# pip install sentencepiece
import os

from transformers import GemmaTokenizer, AutoModelForCausalLM

os.environ["CURL_CA_BUNDLE"] = "/home/aryan/Documents/SecurityAppliance_SSL_CA.pem"

tokenizer = GemmaTokenizer.from_pretrained("google/codegemma-7b",token="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")
model = AutoModelForCausalLM.from_pretrained("google/codegemma-7b",token="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")

input_text = "Write me a Python function to calculate the nth fibonacci number."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))