import random,sqlite3, json, os, re
from suppress import suppress_stdout_stderr
from llama_cpp import Llama
import pandas as pd
from MSchema.schema_engine import SchemaEngine
from sqlalchemy import create_engine

def execute_sql(predicted_sql, ground_truth_sql, db_path):
    if predicted_sql is None:
        print("Predicted SQL is None")
        return 0
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

def cot_prompt(schema, evidence, question):
    template = (
        "You are an expert SQL query generator. Follow this step-by-step process to generate an optimized SQLite query:\n\n"
        "1. Schema Analysis:\n"
        "   - List the relevant tables and their key columns\n"
        "   - Identify the relationships between tables\n"
        "   - Note any important constraints\n\n"
        "2. Question Breakdown:\n"
        "   - Core data requirements\n"
        "   - Required filters or conditions\n"
        "   - Needed calculations or aggregations\n"
        "   - Required sorting or limiting\n\n"
        "3. Query Construction:\n"
        "   - Write the query following SQLite syntax\n"
        "   - Ensure all referenced columns exist in the schema\n"
        "   - Include proper joins based on identified relationships\n"
        "   - Apply necessary filters and aggregations\n"
        "   - End the query with a semicolon\n\n"
        "4. Query Review:\n"
        "   - Verify all table/column references\n"
        "   - Check join conditions\n"
        "   - Confirm aggregations and groupings\n"
        "   - Validate against requirements\n\n"
        # "Format your response as follows:\n"
        # "SCHEMA ANALYSIS:\n"
        # "[Your analysis here]\n\n"
        # "QUESTION BREAKDOWN:\n"
        # "[Your breakdown here]\n\n"
        # "QUERY:\n"
        # "[Your SQL query here]\n\n"
        # "REVIEW:\n"
        # "[Your review notes here]"
    )

    prompt = (
        f"{template}\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"External Knowledge:\n{evidence}\n\n"
        f"Question: {question}"
    )
    return prompt

def model_predict(model_path, question, schema, temperature, evidence):
    
    prompt = cot_prompt(schema, evidence, question)
        
    with suppress_stdout_stderr():
        llm = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=-1)
        output = llm(
            prompt = prompt,
            max_tokens=800,
            temperature=temperature,
            stop=["This ", "Explanation: ", "Note:", "The", ";", r"\\", r"\end{code}", "Comment:"],
            echo=True
        )
    # response = output['choices'][0]['text']
    response = output['choices'][0]['text']
    match = re.search(r"SELECT .*", response, re.DOTALL | re.IGNORECASE)
    result = match.group(0) if match else "NULL"
    result = result.replace(";", "").replace("```", "").replace("\n", " ")
    return result
    # sql_query = extract_sql_query(response)
    # return sql_query
    
    
def get_schema(db_dir,db_name):
    db_path = os.path.join(db_dir, db_name, db_name + ".sqlite")
    abs_path = os.path.abspath(db_path)
    assert os.path.exists(abs_path)
    db_engine = create_engine(f'sqlite:///{abs_path}')

    schema_engine = SchemaEngine(engine=db_engine, db_name=db_name)
    mschema = schema_engine.mschema
    mschema_str = mschema.to_mschema()
    return mschema_str

def analyse(csv_file, model_path):
    db_dir = "Datasets/bird/databases"
    temperatures = [0.1, 0.5, 1.0]
    result = pd.DataFrame(columns=["Question", "hardness", "Gold", "Prediction_0.1", "Prediction_0.5", "Prediction_1.0", "EX_0.1", "EX_0.5", "EX_1.0"], dtype=object)
    
    with open("Datasets/bird/dev.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    data = [entry for entry in data if entry["difficulty"] == "challenging"]
    
    random.shuffle(data)
    
    data = data[:50]
    
    total_questions = len(data)
    
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
    
    for temp in temperatures:
        ex_percent = (result[f"EX_{temp}"].sum() / total_questions) * 100 if total_questions else 0
        print(f"Temperature={temp}: EX = {ex_percent:.2f}%")
    
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

csv_file = "mschema-bird-challenging-temps.csv"
model_path = "Models/DeepSeek-Coder-V2-Lite-2.4B-Instruct-Q8_0.gguf"

analyse(csv_file, model_path)

metrics_df = calculate_metrics(csv_file)
print('-------------------------------------------------------------------------------')
print(metrics_df)
print('-------------------------------------------------------------------------------')