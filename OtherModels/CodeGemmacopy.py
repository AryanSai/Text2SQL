# pip install sentencepiece
import os

os.environ["CURL_CA_BUNDLE"] = "/home/aryan/Documents/SecurityAppliance_SSL_CA.pem"
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="google/codegemma-7b",token="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")

output = pipe("Write me a Python function to calculate the nth fibonacci number.")

print(output)