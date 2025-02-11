import random,sqlite3, json, os, re
from suppress import suppress_stdout_stderr
from llama_cpp import Llama
import pandas as pd
from MSchema.schema_engine import SchemaEngine
from sqlalchemy import create_engine

def execute_sql(predicted_sql, ground_truth_sql, db_path):
    if predicted_sql == 'NULL':
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
        
        "You are an expert SQL query generator. Follow this step-by-step process to generate an optimized SQLite query. "
        "Think carefully and reason through each step before writing the final query.\n\n"
        
        "### Instructions:\n"
        "1. **Schema Analysis**:\n"
        "   - List all relevant tables and their key columns.\n"
        "   - Identify relationships between tables (e.g., foreign keys).\n"
        "   - Note any constraints or special conditions.\n\n"
        
        "2. **Question Breakdown**:\n"
        "   - Identify the core data requirements (e.g., columns to select).\n"
        "   - Determine any filters or conditions (e.g., WHERE clauses).\n"
        "   - Identify any calculations or aggregations needed (e.g., SUM, AVG).\n"
        "   - Determine if sorting or limiting is required (e.g., ORDER BY, LIMIT).\n\n"
        
        "3. **Query Construction**:\n"
        "   - Write the SQL query step by step, following SQLite syntax.\n"
        "   - Ensure all referenced columns and tables exist in the schema.\n"
        "   - Use proper JOINs based on the identified relationships.\n"
        "   - Apply necessary filters, aggregations, and sorting.\n"
        "   - End the query with a semicolon (`;`).\n\n"
        
        "4. **Query Review**:\n"
        "   - Verify that all table and column references are correct.\n"
        "   - Check that JOIN conditions are accurate.\n"
        "   - Confirm that aggregations and groupings are applied correctly.\n"
        "   - Ensure the query meets all requirements from the question.\n\n"
        
        "Ensure the query is syntactically correct and follows best practices. "
        "Understand the schema provided below. Use the external knowledge provided below to understand the question better."
        "Use only the column names available in the provided database schema. Do not reference columns that do not exist, and ensure you correctly map columns to their respective tables. "
        "### Output Format:\n"
        "Wrap the final SQL query in ```sql``` delimiters. Do not include any additional explanations or notes.\n\n"
        
        "### Examples:\n"
        "Example 1:\n"
        "Question: What is the average salary of employees in the Sales department?\n"
        "Schema Analysis:\n"
        "- Tables: `employees` (columns: `id`, `name`, `salary`, `department_id`), `departments` (columns: `id`, `name`).\n"
        "- Relationship: `employees.department_id` references `departments.id`.\n"
        "Question Breakdown:\n"
        "- Core data: Average salary (`AVG(salary)`).\n"
        "- Filter: Only employees in the Sales department (`departments.name = 'Sales'`).\n"
        "- Join: `employees` and `departments` on `department_id`.\n"
        "Query:\n"
        "```sql\n"
        "SELECT AVG(salary)\n"
        "FROM employees\n"
        "JOIN departments ON employees.department_id = departments.id\n"
        "WHERE departments.name = 'Sales';\n"
        "```\n\n"
        
        "Example 2:\n"
        "Question: How many orders were placed in 2023?\n"
        "Schema Analysis:\n"
        "- Tables: `orders` (columns: `id`, `order_date`, `amount`).\n"
        "Question Breakdown:\n"
        "- Core data: Count of orders (`COUNT(*)`).\n"
        "- Filter: Orders placed in 2023 (`order_date BETWEEN '2023-01-01' AND '2023-12-31'`).\n"
        "Query:\n"
        "```sql\n"
        "SELECT COUNT(*)\n"
        "FROM orders\n"
        "WHERE order_date BETWEEN '2023-01-01' AND '2023-12-31';\n"
        "```\n\n"
        
        "### Your Task:\n"
        "Now, generate the SQL query for the following question:\n"
    )

    prompt = (
        f"{template}\n\n"
        "### Database Schema:\n"
        f"{schema}\n\n"
        "### External Knowledge:\n"
        f"{evidence}\n\n"
        "### Question:\n"
        f"{question}\n\n"
        "### Steps to Follow:\n"
        "1. Perform Schema Analysis.\n"
        "2. Break down the Question.\n"
        "3. Construct the SQL Query.\n"
        "4. Review the Query for accuracy.\n"
        "5. Output the final query in ```sql``` delimiters."
    )
    return prompt

# def cot_prompt(schema, evidence, question):
#     template = (
#         "You are an expert SQL query generator. Follow this step-by-step process to generate an optimized SQLite query:\n\n"
#         "1. Schema Analysis:\n"
#         "   - List the relevant tables and their key columns\n"
#         "   - Identify the relationships between tables\n"
#         "   - Note any important constraints\n\n"
#         "2. Question Breakdown:\n"
#         "   - Core data requirements\n"
#         "   - Required filters or conditions\n"
#         "   - Needed calculations or aggregations\n"
#         "   - Required sorting or limiting\n\n"
#         "3. Query Construction:\n"
#         "   - Write the query following SQLite syntax\n"
#         "   - Ensure all referenced columns exist in the schema\n"
#         "   - Include proper joins based on identified relationships\n"
#         "   - Apply necessary filters and aggregations\n"
#         "   - End the query with a semicolon\n\n"
#         "4. Query Review:\n"
#         "   - Verify all table/column references\n"
#         "   - Check join conditions\n"
#         "   - Confirm aggregations and groupings\n"
#         "   - Validate against requirements\n\n"
#         # "Format your response as follows:\n"
#         # "SCHEMA ANALYSIS:\n"
#         # "[Your analysis here]\n\n"
#         # "QUESTION BREAKDOWN:\n"
#         # "[Your breakdown here]\n\n"
#         # "QUERY:\n"
#         # "[Your SQL query here]\n\n"
#         # "REVIEW:\n"
#         # "[Your review notes here]"
#     )

#     prompt = (
#         f"{template}\n\n"
#         f"Database Schema:\n{schema}\n\n"
#         f"External Knowledge:\n{evidence}\n\n"
#         f"Question: {question}"
#     )
#     return prompt

def model_predict(model_path, question, schema, temperature, evidence):
    prompt = cot_prompt(schema, evidence, question)
    
    try:
        with suppress_stdout_stderr():
            llm = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=-1)
            output = llm(
                prompt=prompt,
                max_tokens=800,
                temperature=temperature,
                # stop=["```", "This ", "Explanation: ", "Note:", "The", ";", r"\\", r"\end{code}", "Comment:"],
                echo=False
            )
        response = output['choices'][0]['text']
        print('---------------------------------------------------------------------------------------------------')
        print(f"Model Response: {response}")  # Debugging: Print the full response
        print('---------------------------------------------------------------------------------------------------')
        
        # Extract SQL query from ```sql``` delimiters or fallback to SELECT pattern
        # match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL | re.IGNORECASE)
        # if not match:
        #     match = re.search(r"SELECT .*", response, re.DOTALL | re.IGNORECASE)
        # result = match.group(0) if match else "NULL"
        # result = re.sub(r"^\s*sql\s+", "", result, flags=re.IGNORECASE)
        # result = result.replace(";", "").replace("```", "").replace("\n", " ")
        # Try to match a SQL code block first
        match = re.search(r"```sql\s*\n(.*?)\n?```", response, re.DOTALL | re.IGNORECASE)
        
        if not match:
            # If no SQL block, extract starting from SELECT
            match = re.search(r"(SELECT\s.*)", response, re.DOTALL | re.IGNORECASE)
    
        result = match.group(1) if match else "NULL"
    
        # Remove unwanted "sql" prefix if present
        result = re.sub(r"^\s*sql\s+", "", result, flags=re.IGNORECASE)
    
        # Clean up unwanted characters
        result = result.replace("```", "").replace(";", "").strip()
    
        # Normalize spaces (remove excessive newlines)
        result = re.sub(r"\s+", " ", result)
        print('---------------------------------------------------------------------------------------------------')
        print(f"Extracted SQL: {result}")  # Debugging: Print the extracted SQL
        print('---------------------------------------------------------------------------------------------------')
    except Exception as e:
        print(f"Error in model prediction: {e}")
        result = "NULL"
    return result
    
    
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

csv_file = "Codestral-mschema-cot.csv"
model_path = "Models/Codestral-22B-v0.1-Q8_0.gguf"

analyse(csv_file, model_path)

metrics_df = calculate_metrics(csv_file)
print('-------------------------------------------------------------------------------')
print(metrics_df)
print('-------------------------------------------------------------------------------')