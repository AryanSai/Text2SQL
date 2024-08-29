import pandas as pd, json

csv_file = 'Results/Codestral22B/Codestral-22B-v0.1-Q8_0_Consistency_Results.csv'
df = pd.read_csv(csv_file)

json_file = 'Datasets/spider_dev_with_hardness.json'
with open(json_file, 'r') as f:
    json_data = json.load(f)

question_to_dbid = {item['question']: item['db_id'] for item in json_data}

df['db_id'] = df['Question'].map(question_to_dbid)

output_csv_file = 'new_csv_file.csv'
df.to_csv(output_csv_file, index=False)

print(f"Updated CSV saved to {output_csv_file}")
