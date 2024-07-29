from huggingface_hub import snapshot_download
from pathlib import Path
import os

os.environ["CURL_CA_BUNDLE"] = "/home/aryan/Documents/SecurityAppliance_SSL_CA.pem"
mistral_models_path = Path.home().joinpath('mistral_models', 'mamba-codestral-7B-v0.1')
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="mistralai/mamba-codestral-7B-v0.1", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path,token="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")
