import sqlite3, json, os, re
from suppress import suppress_stdout_stderr
from llama_cpp import Llama
from evaluation import main
import pandas as pd
from MSchema.schema_engine import SchemaEngine
from sqlalchemy import create_engine

def model_predict(model_path, question, schema, schema_description=""):
    template = (
    "You are an expert SQL query generator. Convert the given question into an optimized SQLite query. "
    "Ensure the query is syntactically correct and follows best practices. "
    "Use only the column names available in the provided database schema. "
    "Do not reference columns that do not exist, and ensure you correctly map columns to their respective tables. "
    "Do not include explanations, comments, or extra text. The query must end with a semicolon (;)."
    )
    if schema_description:
        prompt = (
            f"\n{schema}\n\n"
            f"Schema Description:\n{schema_description}\n\n"
            f"{template}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
    else:
        prompt = (
            f"\n{schema}\n\n"
            f"{template}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        
    with suppress_stdout_stderr():
        llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1)
        output = llm(
            prompt = prompt,
            max_tokens=300,
            temperature=0.1,
            stop=["This ", "Explanation: ", "Note:", "The", ";", r"\\", r"\end{code}", "Comment:"],
            echo=True
        )
    response = output['choices'][0]['text']
    match = re.search(r"SELECT .*", response, re.DOTALL | re.IGNORECASE)
    # if match:
    #     result = match.group(0)
    # else:
    #     print("NULL:", match,"\n")
    #     result = "NULL"
    result = match.group(0) if match else "NULL"
    ###################################################
    if result == "NULL":
        print("Retrying with adjusted parameters...")
    output = llm(prompt=prompt, max_tokens=400, temperature=0.2, stop=[";", "Explanation:", "Note:"])
    response = output['choices'][0]['text']
    match = re.search(r"SELECT .*", response, re.DOTALL | re.IGNORECASE)
    result = match.group(0) if match else "NULL"
    ###################################################
    result = result.replace(";", "").replace("```", "").replace("\n", " ")
    return result

def get_desc():
    with open("concert_singer.txt", "r") as file:
        schema_description = file.read()
    return schema_description

def get_mschema():
    db_name= 'concert_singer'
    db_path = 'Datasets/spider/database/concert_singer/concert_singer.sqlite'
    abs_path = os.path.abspath(db_path)
    assert os.path.exists(abs_path)
    db_engine = create_engine(f'sqlite:///{abs_path}')

    schema_engine = SchemaEngine(engine=db_engine, db_name=db_name)
    mschema = schema_engine.mschema
    mschema_str = mschema.to_mschema()
    return mschema_str

def get_schema():
    db = os.path.join("Datasets/spider/database", "concert_singer", "concert_singer" + ".sqlite")
    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]
    for table in tables:
        cursor.execute(f'PRAGMA table_info("{table}")')
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]
    return schema

def analyse(csv_file,model_path):
    schema = get_mschema()
    result = pd.DataFrame(columns=["Question","Hardness","Gold", "Without Description", "With Description", 
                                   "EM Without", "EX Without", "EM With", "EX With"],dtype=object)
    
    with open("Datasets/spider/dev_with_hardness.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    data = [entry for entry in data if entry["db_id"] == "concert_singer"]

    total_questions = len(data)
    total_em_without, total_ex_without = 0, 0
    total_em_with, total_ex_with = 0, 0
    
    for entry in data:
        question = entry['question']
        gold = entry['query'] + "\t" + entry['db_id']
        
        response_without_desc = model_predict(model_path,question,schema,"")
        if(response_without_desc == "NULL"):
            with open("MistakeBook.txt", "a") as f:
                f.write(question)
            break

        em_without, ex_without = main(gold, response_without_desc, "Datasets/spider/database", "Datasets/spider/tables.json")
        
        print(f"Question: {question}")
        print("Prediction without schema description:", response_without_desc)
        
        response_with_desc = model_predict(model_path,question,schema,get_desc())
        em_with, ex_with = main(gold, response_with_desc, "Datasets/spider/database", "Datasets/spider/tables.json")
        
        print("Prediction with schema description:", response_with_desc)
        print('--------------------------------------------------------------')
        
        em_without = int(bool(em_without))
        ex_without = int(bool(ex_without))
        em_with = int(bool(em_with))
        ex_with = int(bool(ex_with))
        
        result.loc[len(result)] = [question, entry["hardness"], entry['query'], response_without_desc, response_with_desc, em_without, ex_without, em_with, ex_with]

        total_em_without += em_without
        total_ex_without += ex_without
        total_em_with += em_with
        total_ex_with += ex_with
        
    result.to_csv(csv_file, index=False)
    
    em_without_percent = (total_em_without / total_questions) * 100 if total_questions else 0
    ex_without_percent = (total_ex_without / total_questions) * 100 if total_questions else 0
    em_with_percent = (total_em_with / total_questions) * 100 if total_questions else 0
    ex_with_percent = (total_ex_with / total_questions) * 100 if total_questions else 0

    print(f"Total Questions: {total_questions}")
    print(f"EM Without Description: {em_without_percent:.2f}%")
    print(f"EX Without Description: {ex_without_percent:.2f}%")
    print(f"EM With Description: {em_with_percent:.2f}%")
    print(f"EX With Description: {ex_with_percent:.2f}%")

    return result

def calculate_metrics(csv_file):
    df = pd.read_csv(csv_file)
    categories = df['Hardness'].unique()
    metrics = []
    for category in categories:
        subset = df[df['Hardness'] == category]
        total = len(subset)
        
        em_without = subset['EM Without'].sum()
        ex_without = subset['EX Without'].sum()
        em_with = subset['EM With'].sum()
        ex_with = subset['EX With'].sum()
        
        metrics.append({
            "Hardness": category,
            "Total Questions": total,
            "EM Without (%)": (em_without / total) * 100 if total else 0,
            "EX Without (%)": (ex_without / total) * 100 if total else 0,
            "EM With (%)": (em_with / total) * 100 if total else 0,
            "EX With (%)": (ex_with / total) * 100 if total else 0,
        })
    
    return pd.DataFrame(metrics)

#############################################################################################################

csv_file = "mschema-codestral_desc_analysis.csv"
model_path = "Models/Codestral-22B-v0.1-Q8_0.gguf"

analyse(csv_file,model_path)

metrics_df = calculate_metrics(csv_file)
print(metrics_df)
