import sqlite3
import json
import os
import re
import pandas as pd
from suppress import suppress_stdout_stderr
from llama_cpp import Llama

def execute_sql(predicted_sql, ground_truth_sql, db_path):
    """
    Executes predicted and ground truth SQL queries on the given SQLite database and compares results.
    Returns 1 if results match, else 0.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(predicted_sql)
            predicted_res = cursor.fetchall()

            cursor.execute(ground_truth_sql)
            ground_truth_res = cursor.fetchall()

            return int(set(predicted_res) == set(ground_truth_res))

    except sqlite3.Error as e:
        print(f"[SQL Execution Error] {e}")
        return 0

def build_basic_prompt(question, schema, evidence):
    """
    Builds the simple (basic) prompt string for SQL generation.
    """
    template = (
        "You are an expert SQL query generator. Convert the given question into an optimized SQLite query. "
        "Ensure the query is syntactically correct and follows best practices. "
        "Understand the schema provided below. Use the external knowledge provided below to understand the question better. "
        "Use only the column names available in the provided database schema. Do not reference columns that do not exist, and ensure you correctly map columns to their respective tables. "
        "Do not include explanations, comments, or extra text. The query must end with a semicolon (;)."
    )

    prompt = (
        f"{template}\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"External Knowledge:\n{evidence}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    return prompt

def build_advanced_prompt(question, schema, evidence):
    """
    Builds the advanced prompt string that generates step-by-step sub-questions for decomposition.
    """
    template = (
        "You are an expert at breaking down complex SQL generation tasks into simple, logical, step-by-step sub-questions. "
        "Given a natural language question and its corresponding SQL query, your task is to decompose the question into a series of incremental sub-questions. "
        "Each sub-question should help guide a human or a system to build the SQL query in stages. "
        "Do not mention SQL syntax or use code in your response. "
        "Use simple and clear language, and number each sub-question in order. "
        "Do not include explanations, comments, or extra text. Only output the sub-questions."
        "Here are some examples:\n\n"
    
    "Example 1:\n"
    "Database Schema:\n"
    "[Description of the schema for context, or you can leave it blank if not necessary]\n\n"
    "External Knowledge:\n"
    "'Difference between K-12 enrollment and 15-17 enrollment can be computed by `Enrollment (K-12)` - `Enrollment (Ages 5-17)`'\n\n"
    "Original Question:\n"
    "Consider the average difference between K-12 enrollment and 15-17 enrollment of schools that are locally funded, "
    "list the names and DOC type of schools which has a difference above this average.\n\n"
    "SQL Query:\n"
    "SELECT T2.School, T2.DOC FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode "
    "WHERE T2.FundingType = 'Locally funded' "
    "AND (T1.`Enrollment (K-12)` - T1.`Enrollment (Ages 5-17)`) > "
    "(SELECT AVG(T3.`Enrollment (K-12)` - T3.`Enrollment (Ages 5-17)`) "
    "FROM frpm AS T3 INNER JOIN schools AS T4 ON T3.CDSCode = T4.CDSCode "
    "WHERE T4.FundingType = 'Locally funded');\n\n"
    
    "Step-by-Step Sub-Questions:\n"
    "1. What schools are locally funded?\n"
    "2. For each of these schools, what is the difference between their K-12 enrollment and 15-17 enrollment?\n"
    "3. What is the average of these differences across all locally funded schools?\n"
    "4. Which schools have a difference greater than this average?\n"
    "5. What are the names and DOC types of these schools?\n\n"
    
    "Example 2:\n"
    "Database Schema:\n"
    "[...] \n\n"
    "External Knowledge:\n"
    "Non-chartered schools refer to schools whose Charter = 0; "
    "percent of eligible free meal rate for K-12 = `Free Meal Count (K-12)` * 100 / `Enrollment (K-12)`.\n\n"
    "Original Question:\n"
    "What is the total number of non-chartered schools in the county of Los Angeles with a percent (%) of eligible free meals "
    "for grades 1 through 12 that is less than 0.18%?\n\n"
    
    "SQL Query:\n"
    "SELECT COUNT(T2.School) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode "
    "WHERE T2.County = 'Los Angeles' AND T2.Charter = 0 "
    "AND CAST(T1.`Free Meal Count (K-12)` AS REAL) * 100 / T1.`Enrollment (K-12)` < 0.18;\n\n"
    
    "Step-by-Step Sub-Questions:\n"
    "1. Which schools are in the county of Los Angeles?\n"
    "2. Among those, which schools are non-chartered?\n"
    "3. For each of these schools, what is the percentage of eligible free meals for grades K-12?\n"
    "4. Which schools have this percentage less than 0.18%?\n"
    "5. How many such schools are there?\n\n"

    "Example 3:\n"
    "Database Schema:\n"
    "[...] \n\n"
    "External Knowledge:\n"
    "Intermediate/Middle Schools refer to SOC = 62; Unified School refers to DOC = 54; "
    "years between 2009 and 2010 can refer to 'between 1/1/2009 to 12/31/2010'.\n\n"
    
    "Original Question:\n"
    "What are the valid e-mail addresses of the administrator of the school located in the San Bernardino county, "
    "City of San Bernardino City Unified that opened between 1/1/2009 to 12/31/2010 whose school types are public "
    "Intermediate/Middle Schools and Unified Schools?\n\n"
    
    "SQL Query:\n"
    "SELECT T2.AdmEmail1, T2.AdmEmail2 FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode "
    "WHERE T2.County = 'San Bernardino' AND T2.City = 'San Bernardino' AND T2.DOC = 54 "
    "AND strftime('%Y', T2.OpenDate) BETWEEN '2009' AND '2010' AND T2.SOC = 62;\n\n"
    
    "Step-by-Step Sub-Questions:\n"
    "1. What schools are located in San Bernardino county?\n"
    "2. Among these, which are in the city of San Bernardino?\n"
    "3. Which of these are public Intermediate/Middle Schools and Unified Schools?\n"
    "4. Which of these schools opened between 1/1/2009 and 12/31/2010?\n"
    "5. What are the valid administrator email addresses of these schools?\n\n"
    )

    prompt = (
        f"{template}\n\n"
        "Now, perform the same decomposition on the following example.\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"External Knowledge:\n{evidence}\n\n"
        f"Question:\n{question}\n\n"
    )

    return prompt

def model_predict(llm, prompt, extract_sql=False):
    """
    Uses the Llama model to predict the SQL query from the given prompt.
    """
    with suppress_stdout_stderr():
        output = llm(
            prompt=prompt,
            max_tokens=600,
            temperature=0.1,
            stop=["This ", "Explanation: ", "Note:", "The", ";", r"\\", r"\end{code}", "Comment:"],
            echo=False
        )

    response = output['choices'][0]['text']

    if extract_sql:
        # Extract SQL query from response
        match = re.search(r"SELECT .*", response, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(0).replace(";", "").replace("```", "").replace("\n", " ")
        else:
            sql = "NULL"
        return sql

    return response.strip()

def get_schema(db_dir, db_id):
    """
    Extracts schema from the SQLite database and returns it as a dictionary.
    """
    db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
    schema = {}

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0].lower() for table in cursor.fetchall()]

        for table in tables:
            cursor.execute(f'PRAGMA table_info("{table}")')
            columns = [col[1].lower() for col in cursor.fetchall()]
            schema[table] = columns

    return schema

def analyse(dataset_file, model_path, output_csv):
    """
    Runs analysis on a dataset by predicting SQL queries using both basic and advanced prompts.
    Evaluates execution accuracy for both. Saves results to a CSV.
    """
    db_dir = "Datasets/bird/databases"

    with open(dataset_file, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    # data = [entry for entry in dataset if entry['difficulty'] == 'challenging']
    # data = [entry for entry in dataset if entry['difficulty'] == 'moderate']
    # data = [entry for entry in dataset if entry['difficulty'] == 'simple']
    
    with suppress_stdout_stderr():
        llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1, device=0)

    results = []

    for entry in dataset:
        question = entry['question']
        ground_truth_sql = entry['query']
        db_id = entry['db_id']
        evidence = entry['evidence']

        schema = get_schema(db_dir, db_id)
        schema_str = json.dumps(schema, indent=2)

        # Generate prompts
        basic_prompt = build_basic_prompt(question, schema_str, evidence)
        advanced_prompt = build_advanced_prompt(question, schema_str, evidence)

        # Get predictions
        basic_predicted_sql = model_predict(llm, basic_prompt, extract_sql=True)
        advanced_predicted_sql = model_predict(llm, advanced_prompt, extract_sql=True)

        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

        # Execute SQL and compare results
        basic_ex_result = execute_sql(basic_predicted_sql, ground_truth_sql, db_path)
        advanced_ex_result = execute_sql(advanced_predicted_sql, ground_truth_sql, db_path)

        print("=" * 80)
        print(f"Question: {question}")
        print(f"[Basic Prompt] Predicted SQL: {basic_predicted_sql}")
        print(f"[Basic Prompt] Execution Match: {'Yes' if basic_ex_result else 'No'}")
        print(f"[Advanced Prompt] Predicted SQL: {advanced_predicted_sql}")
        print(f"[Advanced Prompt] Execution Match: {'Yes' if advanced_ex_result else 'No'}")

        results.append({
            "Question": question,
            "difficulty": entry["difficulty"],
            "Ground Truth SQL": ground_truth_sql,
            "Basic Predicted SQL": basic_predicted_sql,
            "Advanced Predicted SQL": advanced_predicted_sql,
            "EX Basic": basic_ex_result,
            "EX Advanced": advanced_ex_result
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    print(f"\n[INFO] Analysis complete. Results saved to '{output_csv}'")
    return results_df

def calculate_metrics(results_csv):
    """
    Calculates execution (EX) accuracy for both basic and advanced prompts from the CSV results.
    """
    df = pd.read_csv(results_csv)
    total_questions = len(df)
    categories = df['difficulty'].unique()
    metrics = []
    
    for category in categories:
        print("Metrics Summary of " + category)
        subset = df[df['difficulty'] == category]
        total = len(subset)
        
        ex_basic_correct = subset["EX Basic"].sum()
        ex_advanced_correct = subset["EX Advanced"].sum()

        ex_basic_percentage = (ex_basic_correct / total) * 100 if total else 0
        ex_advanced_percentage = (ex_advanced_correct / total) * 100 if total else 0

        metrics = {
            "Total Questions": total,
            "EX Basic Count": ex_basic_correct,
            "EX Advanced Count": ex_advanced_correct,
            "EX Basic (%)": ex_basic_percentage,
            "EX Advanced (%)": ex_advanced_percentage
        }

        print("\n" + "-" * 80)
        print("Metrics Summary:")
        print(pd.DataFrame([metrics]))
        print("-" * 80)
    
    ex_basic_correct = df["EX Basic"].sum()
    ex_advanced_correct = df["EX Advanced"].sum()
    ex_basic_percentage = (ex_basic_correct / total_questions) * 100 if total_questions else 0
    ex_advanced_percentage = (ex_advanced_correct / total_questions) * 100 if total_questions else 0
    metrics = {
        "Total Questions": total_questions,
        "EX Basic Count": ex_basic_correct,
        "EX Advanced Count": ex_advanced_correct,
        "EX Basic (%)": ex_basic_percentage,
        "EX Advanced (%)": ex_advanced_percentage
    }
    print("\n" + "-" * 80)
    print("Metrics Summary:")
    print(pd.DataFrame([metrics]))
    print("-" * 80) 

if __name__ == "__main__":
    dataset_file = "Datasets/bird/dev.json"
    model_path = "Models/qwen2.5-coder-7b-instruct-q8_0.gguf"
    output_csv = "SoQ-Qwen.csv"

    analyse(dataset_file, model_path, output_csv)
    calculate_metrics(output_csv)
