from mistral_inference import MistralChat

import mistral_inference
print(dir(mistral_inference))


# Load the model
model_path = "$HOME/mistral_models/mamba-codestral-7B-v0.1"
model = MistralChat(model_path)

# Generate a response
response = model.generate(prompt="Your prompt here", max_tokens=256)
print(response)
