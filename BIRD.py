import random
import sqlite3, json, os, re
from suppress import suppress_stdout_stderr
from llama_cpp import Llama
import pandas as pd

def execute_sql(predicted_sql, ground_truth_sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        
        cursor.execute(ground_truth_sql)
        ground_truth_res = cursor.fetchall()
        res = 1 if set(predicted_res) == set(ground_truth_res) else 0
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        res = 0  # Return 0 in case of an error
    finally:
        conn.close() 
    return res

def model_predict(model_path, question, schema, temperature, evidence):
    template = (
    "You are an expert SQL query generator. Convert the given question into an optimized SQLite query. "
    "Ensure the query is syntactically correct and follows best practices. "
    "Understand the schema provided below. Use the external knowledge provided below to understand the question better."
    "Use only the column names available in the provided database schema. Do not reference columns that do not exist, and ensure you correctly map columns to their respective tables. "
    "Do not include explanations, comments, or extra text. The query must end with a semicolon (;)."
    )
    
    prompt = (
        f"Database Schema:\n{schema}\n\n"
        f"{template}\n\n"
        f"External Knowledge:\n{evidence}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
        
    with suppress_stdout_stderr():
        llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1,device=0)
        output = llm(
            prompt = prompt,
            max_tokens=300,
            temperature=temperature,
            stop=["This ", "Explanation: ", "Note:", "The", ";", r"\\", r"\end{code}", "Comment:"],
            echo=True
        )
    response = output['choices'][0]['text']
    match = re.search(r"SELECT .*", response, re.DOTALL | re.IGNORECASE)
    result = match.group(0) if match else "NULL"
    result = result.replace(";", "").replace("```", "").replace("\n", " ")
    return result
    
def get_schema(db_dir, db):
    db_path = os.path.join(db_dir, db, db + ".sqlite")
    schema = {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]
    for table in tables:
        cursor.execute(f'PRAGMA table_info("{table}")')
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]
    conn.close()
    return schema

def analyse(csv_file, model_path):
    db_dir = "Datasets/bird/databases"
    temperatures = [0.1, 0.5, 1.0]
    result = pd.DataFrame(columns=["Question", "hardness", "Gold", "Prediction_0.1", "Prediction_0.5", "Prediction_1.0", "EX_0.1", "EX_0.5", "EX_1.0"], dtype=object)
    
    with open("Datasets/bird/dev.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    
    for entry in data:
        schema = get_schema(db_dir, entry['db_id']) 
        question = entry['question']
        predictions = {}
        ex_results = {}
        
        for temp in temperatures:
            predictions[f"Prediction_{temp}"] = model_predict(model_path, question, schema, temp, entry['evidence'])
            ex_results[f"EX_{temp}"] = execute_sql(predictions[f"Prediction_{temp}"], entry['query'], os.path.join(db_dir, entry['db_id'], entry['db_id'] + ".sqlite"))
        
        print(f"Question: {question}")
        for temp in temperatures:
            print(f"Prediction (Temperature={temp}): {predictions[f'Prediction_{temp}']}")
        
        result.loc[len(result)] = [question, entry["difficulty"], entry['query'], 
                                   predictions["Prediction_0.1"], predictions["Prediction_0.5"], predictions["Prediction_1.0"],
                                   ex_results["EX_0.1"], ex_results["EX_0.5"], ex_results["EX_1.0"]]

    result.to_csv(csv_file, index=False)
    return result

def calculate_metrics(csv_file):
    df = pd.read_csv(csv_file)
    categories = df['hardness'].unique()
    metrics = []
    for category in categories:
        subset = df[df['hardness'] == category]
        total = len(subset)
        metrics_entry = {"hardness": category, "Total Questions": total}
        
        for temp in [0.1, 0.5, 1.0]:
            ex = subset[f"EX_{temp}"].sum()
            metrics_entry[f"EX_{temp} (%)"] = (ex / total) * 100 if total else 0
        
        metrics.append(metrics_entry)
    
    return pd.DataFrame(metrics)

#############################################################################################################

csv_file = "qwen-bird-full.csv"
model_path = "Models/qwen2.5-coder-7b-instruct-q8_0.gguf"

analyse(csv_file, model_path)

metrics_df = calculate_metrics(csv_file)
print('-------------------------------------------------------------------------------')
print(metrics_df)
print('-------------------------------------------------------------------------------')