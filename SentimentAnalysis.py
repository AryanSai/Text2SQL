import os

from transformers import pipeline

os.environ["CURL_CA_BUNDLE"] = "/home/aryan/Documents/SecurityAppliance_SSL_CA.pem"

print(pipeline('sentiment-analysis')('we love you'))