import argparse, torch
import seaborn as sns
import re,json,pandas as pd,sqlite3,os,sys
import matplotlib.pyplot as plt
from llama_cpp import Llama
from evaluation import main
# from list_generator import pick_difficulty_lists
import wandb
wandb.login(key = 'b58371874ad31931082450505a758fce636f6d3f')

class suppress_stdout_stderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup    = sys.stdout.fileno()
        self.old_stderr_fileno_undup    = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )

        sys.stdout = self.outnull_file        
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):        
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )

        os.close ( self.old_stdout_fileno )
        os.close ( self.old_stderr_fileno )

        self.outnull_file.close()
        self.errnull_file.close()

def fetchSchema(db):
    """
    Get database's schema, which is a dict with table name as query
    and list of column names as value
    :param db: database path
    :return: schema dict
    """

    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # fetch table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]

    # fetch table info
    for table in tables:
        cursor.execute('PRAGMA table_info("{}")'.format(table))
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

    st = ""
    for query in schema:
        st += query + " : " + str(schema[query]) + "\n"
    return st

def model_predict(question):
    template = "Question: Convert the following text to an SQLite query and end the query with a semi-colon(;). Please provide only the query without any explanation: " 
    with suppress_stdout_stderr():
        llm = Llama(model_path = model_path, n_ctx=2048,n_gpu_layers=-1)
        output = llm(
            prompt = template + question + "\nAnswer:",
            max_tokens=300,
            temperature=0.1,
            stop=["This ", "Explanation: ", "Note:", "The", ";", r"\\", r"\end{code}", "Comment:"],
            echo=True
        )
            
    responseonse = output['choices'][0]['text']
    # print("\nOutput After: ", output)
    match = re.search("SELECT .*",responseonse,re.DOTALL | re.IGNORECASE)
    str = ""
    if(match):  
        str = match.group(0)
    else:  str = "NULL"
    str = str.replace(";","")
    str = str.replace("```","")
    str = str.replace("\n",' ')
    # print("\nOutput After after: ", str)
    return str

def get_difficulty_index(difficulty_level):
    if difficulty_level == 'simple':
        return 0
    elif difficulty_level == 'moderate':
        return 1
    elif difficulty_level == 'challenging':
        return 2
    else:
        raise ValueError("Unknown difficulty level")

def create_labeled_dataframe(matrix):
    # """Creates a DataFrame with formatted labels for each cell."""
    data = [[format_cell_value(matrix, i, j) for j in range(2)] for i in range(2)]
    return pd.DataFrame(data, columns=["0", "1"], index=["0", "1"])

def format_cell_value(matrix, i, j):
    # """Formats the value of a cell to include the counts of each difficulty level."""
    values = matrix[i][j]
    return f"Overall: {sum(values)}\nSimple: {values[0]}\nModerate: {values[1]}\nChallenging: {values[2]}"

def plot_heatmap(title,df,labels,measure_name):
    plt.figure(figsize=(10, 6)) 
    plot = sns.heatmap(df, cmap="Blues", annot=labels, fmt="", annot_kws={"size": 12})
    plt.title(title) 
    plt.suptitle('n = ' + str(repeat), fontsize=10)
    plot.figure.savefig(results_path + measure_name +"_BIRD.png")  
    plt.close()

def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    final_list = data.get('final_list', [])
    repeat = data.get('repeat', 1)
    return final_list, repeat

if __name__ == "__main__":
    
    wandb.init('consistency-bird')
    
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(0)
        free_memory =  gpu_properties.total_memory - torch.cuda.memory_allocated(0)
    else:
        free_memory =  0
        
    print('===========================================================================================')
    print(f"Free GPU Memory: {free_memory / (1024 ** 3):.2f} GB")
    # print(f"Device name: {torch.cuda.get_device_name(0)}")
    print('===========================================================================================')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', dest='modelpath', type=str, required=True)
    parser.add_argument('--dataset', dest='dataset', type=str, required=True)
    parser.add_argument('--dbdir', dest='dbdir', type=str, required=True)
    parser.add_argument('--dbtable', dest='dbtable', type=str, required=True)
    parser.add_argument('--inputjson', dest='inputjson', type=str, required=True)
    parser.add_argument('--output', dest='output', type=str, required=True)
    parser.add_argument('--exact_threshold', dest='exact_threshold', type=float, required=True)
    parser.add_argument('--exec_threshold', dest='exec_threshold', type=float, required=True)
    parser.add_argument('--consistency_threshold', dest='consistency_threshold', type=float, required=True)
    
    args = parser.parse_args()
    
    model_path = args.modelpath
    dataset = args.dataset
    db_dir = args.dbdir
    table = args.dbtable
    input_json = args.inputjson
    results_path = args.output
    threshold_for_execution = args.exec_threshold
    threshold_for_exact = args.exact_threshold
    threshold_for_consistency = args.consistency_threshold
    
    model_name = model_path.split('/')[-1].split('.gguf')[0]
    results_path = results_path + model_name

    file = open(dataset)
    data = json.load(file)
    
    # number_of_samples = divide_samples(50,20,30,50) #percentage of samples for each of the difficulty categories
    # number_of_samples =[5,5,5]
    # final_list = pick_difficulty_lists(dataset,number_of_samples,repeat)

    input_list, repeat = load_json(input_json)

    result = pd.DataFrame(columns = ["Qid","Question","Db_Id","Difficulty","Query Gold","Repeated","Query Generated", "Most Frequent Query"])
    
    for index in input_list:
        schemaPath = db_dir+"/{}/{}.sqlite".format(data[index]['db_id'],data[index]['db_id'])
        schema = fetchSchema(schemaPath)
        response = model_predict(data[index]['question']+ schema)
        print("\nPrediction: ",response)
        
        if not result[result['Qid'] == index].empty:    # already exists
            row_index = result[result['Qid'] == index].index[0]
            result.loc[row_index,'Query Generated'] += "\n" + response
        else:
            curr = [index , data[index]['question'], data[index]['db_id'], data[index]['difficulty'],data[index]['query'],repeat,response,""]
            result.loc[len(result)] = curr

    cm_exact = [[[0, 0, 0] for _ in range(2)] for _ in range(2)]
    cm_exec = [[[0, 0, 0] for _ in range(2)] for _ in range(2)]
    cm_exact_and_exec = [[[0, 0, 0] for _ in range(2)] for _ in range(2)]

    for index, row in result.iterrows():
        map = {}
        exact, exec, i, j = (0,)*4
        qid = row['Qid']
        predicted_queries = row['Query Generated'].split("\n")  # multiple predictions

        # counting the frequencies of the predictions
        for r in predicted_queries:  # result will look like : {'SELECT * FROM tablename': 5}
            if r in map:
                map[r] += 1
            else:
                map[r] = 1

        for query in map:
            gold = row['Query Gold'] + "\t" + data[qid]['db_id']
            pred = query
            m, e = main(gold, pred, db_dir, table)  # returns the exact match and execution accuracy
            
            print(f"Query: {pred}\nExact Match: {m}, Execution: {e}")
            
            if m:
                exact += map[query]
            if e:
                exec += map[query]

        querymax = max(map, key=lambda k: map[k])  # max frequency query
        result.at[index, 'Most Frequent Query'] = querymax
        
        # if more than the threshold, then they are consistent, hence you go to the second row
        # determine consistency
        if map[querymax] / repeat > threshold_for_consistency:
            i = 1
        else:
            i = 0

        exact = exact / repeat
        exec = exec / repeat

        difficulty_index = get_difficulty_index(row['Difficulty'])

        if exact > threshold_for_exact:
            j = 1
        else:
            j = 0
        cm_exact[i][j][difficulty_index] += 1

        if exec > threshold_for_execution:
            j = 1
        else:
            j = 0
        cm_exec[i][j][difficulty_index] += 1

        if exec > threshold_for_execution and exact > threshold_for_exact:
            j = 1
        else:
            j = 0
        cm_exact_and_exec[i][j][difficulty_index] += 1
        
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

        df_cm_exact_and_exec_numeric = pd.DataFrame(
            [[sum(cm_exact_and_exec[i][j]) for j in range(2)] for i in range(2)],
            columns=["0", "1"],
            index=["0", "1"]
        )
        
        #dataFrames with formatted labels
        df_cm_exact_labels = create_labeled_dataframe(cm_exact)
        df_cm_exec_labels = create_labeled_dataframe(cm_exec)
        df_cm_exact_and_exec_labels = create_labeled_dataframe(cm_exact_and_exec)

        # plot the heatmaps
        plot_heatmap(model_name+'_Consistency on Exact Measure',df_cm_exact_numeric,df_cm_exact_labels,"_Exact_Measure")
        plot_heatmap(model_name+'_Consistency on Execution Measure',df_cm_exec_numeric,df_cm_exec_labels,"_Execution_Measure")
        plot_heatmap(model_name+'_Consistency on Both Exact and Execution Measure',df_cm_exact_and_exec_numeric,df_cm_exact_and_exec_labels,"_Exact_And_Execution_Measure")

    result.to_csv(results_path + model_name+"_Consistency_Results_BIRD.csv")
    
    print('===========================================================================================')
    print("Finished Consistency Metric Evaluation")
    wandb.finish()