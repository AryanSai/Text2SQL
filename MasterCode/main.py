from llama_cpp import Llama
import json
import re
import pandas as pd
import torch

# Verify GPU availability
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")


model_name = "Meta-Llama-3.1-8B-Instruct-IQ3_M"
csv_file_name = "Results/" + model_name + ".csv"
llm = Llama(model_path="Models/Meta-Llama-3.1-8B-Instruct-IQ3_M.gguf",n_ctx=2048, n_gpu_layers=-1)

print("model loaded")

f = open("Datasets/spider/dev.json")
data = json.load(f)

df = pd.DataFrame(columns = ["Question","Query Gold","Query Generated"])

template = "Question: Convert the following text statement to sqlite query: " 
i=1
for item in data[:2]:
    print(i)
    try:
        temp = [item['question'],item['query']]
        print(item['question'])
        output = llm(
            # "Question: Convert the following text statement to sqlite query : display all names of students whose marks are above average of all marks of students. Answer:",
            prompt= template + item['question'] + "\nAnswer:",
            # max_tokens=300,
            temperature=0.1,
            stop=["Explanation:"],  # for llama2-7b
            echo=True
        )
        # print(json.dumps(output,indent=2))
        response = output["choices"][0]["text"]
        print(response + "\n")
        match = re.search("SELECT .*",response,re.DOTALL)
        if(match):  
            print("match")
            temp.append(match.group(0))
        else:   temp.append("NULL")
        df.loc[len(df)] = temp
        i+=1
    except:
        print('errrrrr')
        i+=1
        continue
df.to_csv(csv_file_name)