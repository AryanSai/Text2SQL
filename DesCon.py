import random,sqlite3, json, os, re
from suppress import suppress_stdout_stderr
from llama_cpp import Llama
from evaluation import main
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_hardness_index(hardness_level):
    if hardness_level == 'easy':
        return 0
    elif hardness_level == 'medium':
        return 1
    elif hardness_level == 'hard':
        return 2
    elif hardness_level == 'extra':
        return 3
    else:
        raise ValueError("Unknown hardness level")

def create_labeled_dataframe(matrix):
    # """Creates a DataFrame with formatted labels for each cell."""
    data = [[format_cell_value(matrix, i, j) for j in range(2)] for i in range(2)]
    return pd.DataFrame(data, columns=["0", "1"], index=["0", "1"])

def format_cell_value(matrix, i, j):
    values = matrix[i][j]
    return (f"Overall: {sum(values)}\nEasy: {values[0]}\nMedium: {values[1]}"
            f"\nHard: {values[2]}\nExtra Hard: {values[3]}")

def plot_heatmap(title,df,labels):
    plt.figure(figsize=(10, 6)) 
    plot = sns.heatmap(df, cmap="Blues", annot=labels, fmt="", annot_kws={"size": 12})
    plt.title(title) 
    plt.suptitle('n = 7 ', fontsize=10)
    plot.figure.savefig(title +"_SPIDER.png")  
    plt.close()
    
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
            f"Database Schema:\n{schema}\n\n"
            f"Schema Description:\n{schema_description}\n\n"
            f"{template}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
    else:
        prompt = (
            f"Database Schema:\n{schema}\n\n"
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
    result = match.group(0) if match else "NULL"
    result = result.replace(";", "").replace("```", "").replace("\n", " ")
    return result

def get_desc():
    with open("concert_singer.txt", "r") as file:
        schema_description = file.read()
    return schema_description
    
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

def calculate_metrics(csv_file):
    df = pd.read_csv(csv_file)
    categories = df['hardness'].unique()
    metrics = []
    for category in categories:
        subset = df[df['hardness'] == category]
        total = len(subset)
        
        em_without = subset['EM Without'].sum()
        ex_without = subset['EX Without'].sum()
        em_with = subset['EM With'].sum()
        ex_with = subset['EX With'].sum()
        
        metrics.append({
            "hardness": category,
            "Total Questions": total,
            "EM Without (%)": (em_without / total) * 100 if total else 0,
            "EX Without (%)": (ex_without / total) * 100 if total else 0,
            "EM With (%)": (em_with / total) * 100 if total else 0,
            "EX With (%)": (ex_with / total) * 100 if total else 0,
        })
    
    return pd.DataFrame(metrics)

def consistency_analysis(csv_file,model_path):
    db_dir = "Datasets/spider/database"
    table = "Datasets/spider/tables.json"
    threshold_for_execution,threshold_for_exact,threshold_for_consistency = 0.5, 0.5, 0.5
    repeat = 7  
    model_name = model_path.split('/')[-1].split('.gguf')[0]
    
    with open("Datasets/spider/dev_with_hardness.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    data = [entry for entry in data if entry["db_id"] == "concert_singer"]
    total_questions = len(data) 
    
    data = data * repeat
    random.shuffle(data)
    
    total_em_without, total_ex_without, total_em_with, total_ex_with = 0, 0, 0, 0
    
    schema = get_schema()
    result = pd.DataFrame(columns=["Question","Db_Id","hardness","Gold","Repeated", "Without Description", "With Description", 
                                   "EM Without", "EX Without", "EM With", "EX With", 'Without - Most Frequent Query', 
                                   "With - Most Frequent Query","Consistency-EM-Without","Consistency-EX-Without",
                                   "Consistency-EM-With","Consistency-EX-With" ],dtype=object)
    
    for entry in data:
        question = entry["question"]
        gold = entry['query'] + "\t" + entry['db_id']
        
        response_without_desc = model_predict(model_path,question,schema,"")
        _, _ = main(gold, response_without_desc, db_dir, table)
        
        print(f"Question: {question}")
        print("Prediction without schema description:", response_without_desc)
        
        if entry["question"] in result["Question"].values:
            row_index = result.index[result["Question"] == entry["question"]].tolist()[0]
            result.loc[row_index,'Without Description'] += "\n" + response_without_desc
        else:
            result.loc[len(result)] = [question,entry['db_id'], entry["hardness"], entry['query'], repeat, response_without_desc, "", 0, 0, 0, 0,"","",0,0,0,0]
    
        response_with_desc = model_predict(model_path,question,schema,get_desc())
        _, _ = main(gold, response_with_desc, db_dir, table)
        
        print("Prediction with schema description:", response_with_desc)
        print('--------------------------------------------------------------')
        
        row_index = result.index[result["Question"] == entry["question"]].tolist()[0]
        if result.loc[row_index, 'With Description'] == "":
            result.loc[row_index, 'With Description'] = response_with_desc
        else:
            result.loc[row_index, 'With Description'] += "\n" + response_with_desc
        
    cm_exact = [[[0, 0, 0,0] for _ in range(2)] for _ in range(2)]
    cm_exec = [[[0, 0, 0,0] for _ in range(2)] for _ in range(2)]

    for index, row in result.iterrows():
        without_map = {}
        exact, exec, i, j = (0,)*4
        without_predicted_queries = row['Without Description'].split("\n")  # multiple predictions

        # counting the frequencies of the predictions
        for r in without_predicted_queries:  # result will look like : {'SELECT * FROM tablename': 5}
            if r in without_map:
                without_map[r] += 1
            else:
                without_map[r] = 1

        for query in without_map:
            gold = row['Gold'] + "\t" + row['Db_Id']
            pred = query
            m, e = main(gold, pred, db_dir, table)  # returns the exact match and execution accuracy
            
            print(f"Query: {pred}\nExact Match: {m}, Execution: {e}")
            
            if m:
                exact += without_map[query]
            if e:
                exec += without_map[query]

        querymax = max(without_map, key=lambda k: without_map[k])  # max frequency query
        m, e = main(gold, querymax, db_dir, table)
        result.at[index, 'Without - Most Frequent Query'] = querymax
        em_without = int(bool(m))
        ex_without = int(bool(e))
        result.at[index, "EM Without"] = em_without
        result.at[index, "EX Without"] = ex_without
        total_em_without += em_without
        total_ex_without += ex_without
        
        if without_map[querymax] / repeat > threshold_for_consistency:
            i = 1
        else:
            i = 0

        exact = exact / repeat
        exec = exec / repeat

        hardness_index = get_hardness_index(row['hardness'])

        if exact > threshold_for_exact:
            result.at[index, "Consistency-EM-Without"] = 1
            j = 1
        else:
            result.at[index, "Consistency-EM-Without"] = 0
            j = 0
        cm_exact[i][j][hardness_index] += 1

        if exec > threshold_for_execution:
            result.at[index, "Consistency-EX-Without"] = 1
            j = 1
        else:
            result.at[index, "Consistency-EX-Without"] = 0
            j = 0
        cm_exec[i][j][hardness_index] += 1
        
        df_cm_exact_numeric = pd.DataFrame(
            [[sum(cm_exact[i][j]) for j in range(2)] for i in range(2)],
            columns=["0", "1"],
            index=["0", "1"]
        )

        df_cm_exec_numeric = pd.DataFrame(
            [[sum(cm_exec[i][j]) for j in range(2)] for i in range(2)],
            columns=["0", "1"],
            index=["0", "1"]
        )
        
        df_cm_exact_labels = create_labeled_dataframe(cm_exact)
        df_cm_exec_labels = create_labeled_dataframe(cm_exec)

        plot_heatmap("Desc-Without-"+model_name+"_Consistency on Exact Measure",df_cm_exact_numeric,df_cm_exact_labels)
        plot_heatmap("Desc-Without-"+model_name+"_Consistency on Execution Measure",df_cm_exec_numeric,df_cm_exec_labels)
        
        ######################################################################################################
        
        with_map = {}
        exact, exec, i, j = (0,)*4
        with_predicted_queries = row['With Description'].split("\n")  # multiple predictions
        
         # counting the frequencies of the predictions
        for r in with_predicted_queries:  # result will look like : {'SELECT * FROM tablename': 5}
            if r in with_map:
                with_map[r] += 1
            else:
                with_map[r] = 1

        for query in with_map:
            gold = row['Gold'] + "\t" + row['Db_Id']
            pred = query
            m, e = main(gold, pred, db_dir, table)  # returns the exact match and execution accuracy
            
            print(f"Query: {pred}\nExact Match: {m}, Execution: {e}")
            
            if m:
                exact += with_map[query]
            if e:
                exec += with_map[query]

        querymax = max(with_map, key=lambda k: with_map[k])  # max frequency query
        m, e = main(gold, querymax, db_dir, table)
        result.at[index, 'With - Most Frequent Query'] = querymax
        em_with = int(bool(m))
        ex_with = int(bool(e))
        result.at[index, "EM With"] = em_with
        result.at[index, "EX With"] = ex_with
        total_em_with += em_with
        total_ex_with += ex_with
        
        if with_map[querymax] / repeat > threshold_for_consistency:
            i = 1
        else:
            i = 0

        exact = exact / repeat
        exec = exec / repeat

        hardness_index = get_hardness_index(row['hardness'])

        if exact > threshold_for_exact:
            result.at[index, "Consistency-EM-With"] = 1
            j = 1
        else:
            result.at[index, "Consistency-EM-With"] = 0
            j = 0
        cm_exact[i][j][hardness_index] += 1

        if exec > threshold_for_execution:
            result.at[index, "Consistency-EX-With"] = 1
            j = 1
        else:
            result.at[index, "Consistency-EX-With"] = 0
            j = 0
        cm_exec[i][j][hardness_index] += 1
        
        df_cm_exact_numeric = pd.DataFrame(
            [[sum(cm_exact[i][j]) for j in range(2)] for i in range(2)],
            columns=["0", "1"],
            index=["0", "1"]
        )

        df_cm_exec_numeric = pd.DataFrame(
            [[sum(cm_exec[i][j]) for j in range(2)] for i in range(2)],
            columns=["0", "1"],
            index=["0", "1"]
        )
        
        df_cm_exact_labels = create_labeled_dataframe(cm_exact)
        df_cm_exec_labels = create_labeled_dataframe(cm_exec)

        plot_heatmap("Desc-With-"+model_name+"_Consistency on Exact Measure",df_cm_exact_numeric,df_cm_exact_labels)
        plot_heatmap("Desc-With-"+model_name+"_Consistency on Execution Measure",df_cm_exec_numeric,df_cm_exec_labels)
    
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

    print('===========================================================================================')
    print("Finished Consistency Metric Evaluation")

#############################################################################################################

csv_file = "consistencydeepseek_desc_analysis.csv"
model_path = "Models/DeepSeek-Coder-V2-Lite-2.4B-Instruct-Q8_0.gguf"
consistency_analysis(csv_file,model_path)

metrics_df = calculate_metrics(csv_file)
print(metrics_df)
