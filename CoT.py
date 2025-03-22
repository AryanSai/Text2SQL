import sqlite3,json,os,re
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


def build_prompt(question, schema, evidence):
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

def model_predict(llm, question, schema, evidence):
    prompt = build_prompt(question, schema, evidence)
    try:
        with suppress_stdout_stderr():
            output = llm(
                prompt=prompt,
                max_tokens=600,
                temperature=0.1,
                stop=["```", "This ", "Explanation: ", "Note:", "The", ";", r"\\", r"\end{code}", "Comment:"],
                echo=False
            )
        print('---------------------------------------------------------------------------------------------------')
        print(f"Model Output: {output}")  # Debugging: Print the full response
        print('---------------------------------------------------------------------------------------------------')
        response = output['choices'][0]['text']
        print('---------------------------------------------------------------------------------------------------')
        print(f"Model Response: {response}")  # Debugging: Print the full response
        print('---------------------------------------------------------------------------------------------------')
        
        match = re.search(r"```sql\s*\n(.*?)\n?```", response, re.DOTALL | re.IGNORECASE)
        
        if not match:
            match = re.search(r"(SELECT\s.*)", response, re.DOTALL | re.IGNORECASE)
    
        result = match.group(1) if match else "NULL"
    
        result = re.sub(r"^\s*sql\s+", "", result, flags=re.IGNORECASE)
    
        result = result.replace("```", "").replace(";", "").strip()
    
        result = re.sub(r"\s+", " ", result)
        print('---------------------------------------------------------------------------------------------------')
        print(f"Extracted SQL: {result}")  # Debugging: Print the extracted SQL
        print('---------------------------------------------------------------------------------------------------')
    except Exception as e:
        print(f"Error in model prediction: {e}")
        result = "NULL"
    return result

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

    data = [entry for entry in dataset if entry['difficulty'] == 'challenging']
    # data = [entry for entry in dataset if entry['difficulty'] == 'moderate']
    # data = [entry for entry in dataset if entry['difficulty'] == 'simple']
    
    with suppress_stdout_stderr():
        llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1, device=1)

    results = []

    for entry in data:
        question = entry['question']
        ground_truth_sql = entry['query']
        db_id = entry['db_id']
        evidence = entry['evidence']

        schema = get_schema(db_dir, db_id)
        schema_str = json.dumps(schema, indent=2)

        predicted_sql = model_predict(llm, question, schema_str, evidence)

        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

        ex_result = execute_sql(predicted_sql, ground_truth_sql, db_path)

        print("=" * 80)
        print(f"Question: {question}")
        print(f"Predicted SQL: {predicted_sql}")
        print(f"Execution Match: {'Yes' if ex_result else 'No'}")

        results.append({
            "Question": question,
            "Difficulty": entry["difficulty"],
            "Ground Truth SQL": ground_truth_sql,
            "Predicted SQL": predicted_sql,
            "EX": ex_result
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

    ex_correct = df["EX"].sum()

    ex_percentage = (ex_correct / total_questions) * 100 if total_questions else 0

    metrics = {
        "Total Questions": total_questions,
        "EX Count": ex_correct,
        "EX (%)": ex_percentage
    }

    print("\n" + "-" * 80)
    print("Metrics Summary:")
    print(pd.DataFrame([metrics]))
    print("-" * 80)

    return metrics

if __name__ == "__main__":
    dataset_file = "Datasets/bird/dev.json"
    model_path = "Models/qwen2.5-coder-7b-instruct-q8_0.gguf"
    output_csv = "cot-qwen.csv"

    analyse(dataset_file, model_path, output_csv)
    calculate_metrics(output_csv)