import seaborn as sns
import re,json,random,pandas as pd,sqlite3,os,sys
import matplotlib.pyplot as plt
from llama_cpp import Llama
from evaluation import main

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
        cursor.execute("PRAGMA table_info({})".format(table))
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

    st = ""
    for query in schema:
        st += query + " : " + str(schema[query]) + "\n"
    return st

def model_predict(question):
    template = "Question: Convert the following text to an SQLite query and end the query with a semi-colon(;). Please provide only the query without any explanation: " 
    with suppress_stdout_stderr():
        llm = Llama(model_path = model_path, n_ctx=2048)
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

def pick_difficulty_lists(dataset, number_of_samples,repeat):
    with open(dataset, 'r') as f:
        data = json.load(f)
    
    easy = []
    medium = []
    hard = []

    for i, entry in enumerate(data):
        if entry.get("difficulty") == "simple":
            easy.append(i)
        elif entry.get("difficulty") == "moderate":
            medium.append(i)
        elif entry.get("difficulty") == "challenging":
            hard.append(i)

    # randomly sample indices from each category
    easy_sample = random.sample(easy, min(number_of_samples[0], len(easy)))
    medium_sample = random.sample(medium, min(number_of_samples[1], len(medium)))
    hard_sample = random.sample(hard, min(number_of_samples[2], len(hard)))
    
    repeated_easy = [item for item in easy_sample for _ in range(repeat)]
    repeated_medium = [item for item in medium_sample for _ in range(repeat)]
    repeated_hard = [item for item in hard_sample for _ in range(repeat)]

    combined_list = repeated_easy + repeated_medium + repeated_hard
    
    random.shuffle(combined_list)
    
    return combined_list

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
    plot.figure.savefig(results_path + measure_name +dataset_tag+".png")  
    plt.close()

def divide_samples(number_of_samples, simple_pct, moderate_pct, challenging_pct):
    assert simple_pct + moderate_pct + challenging_pct == 100, "Percentages must sum up to 100!"

    simple_samples = int((simple_pct / 100) * number_of_samples)
    moderate_samples = int((moderate_pct / 100) * number_of_samples)
    challenging_samples = number_of_samples - simple_samples - moderate_samples 

    return [simple_samples, moderate_samples, challenging_samples]

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--repeat', dest='repeat', type=int)
    # parser.add_argument('--test', dest='test', type=str)
    # parser.add_argument('--model', dest='model', type=str)
    # parser.add_argument('--schema',dest='schema',type=int)
    # args = parser.parse_args()

    # repeat = args.repeat
    # dataset = args.test
    # model_name = args.model
    # with_schema=args.schema
    
    model_path = 'Models/codegemma-7b-Q8_0.gguf'
    model_name = model_path.split('/')[-1].split('.gguf')[0]
    
    # number_of_samples = divide_samples(50,20,30,50) #percentage of samples for each of the difficulty categories
    number_of_samples =[1,1,1]
    # print("sample size",number_of_samples)
    repeat = 1
    
    # with_schema=False
    with_schema=True
    
    # spider
    db_dir = "Datasets/spider/database"
    table = "Datasets/spider/tables.json"
    dataset_tag = "_SPIDER"
    dataset = 'Datasets/spider_dev_with_hardness.json'
    
    # bird
    # db_dir = "Datasets/bird/databases"
    # table = "Datasets/bird/dev_tables.json"
    # dataset_tag = "_BIRD"
    # dataset = 'Datasets/bird/bird_dev_as_spider.json'
    
    results_path = "Results/"
    results_path= results_path + model_name

    file = open(dataset)
    data = json.load(file)
    
    final_list = pick_difficulty_lists(dataset,number_of_samples,repeat)
    
    result = pd.DataFrame(columns = ["Qid","Question","Db_Id","Difficulty","Query Gold","Repeated","Query Generated"])

    for index in final_list:
        schemaPath = db_dir+"/{}/{}.sqlite".format(data[index]['db_id'],data[index]['db_id'])
        if with_schema: 
            schema = fetchSchema(schemaPath)
        else: schema = ""
        
        response = model_predict(data[index]['question']+ schema)
        print("\nPrediction: ",response)
        
        if not result[result['Qid'] == index].empty:    # already exists
            row_index = result[result['Qid'] == index].index[0]
            result.loc[row_index,'Query Generated'] += "\n" + response
        else:
            curr = [index , data[index]['question'], data[index]['db_id'], data[index]['difficulty'],data[index]['query'],repeat,response]
            result.loc[len(result)] = curr
    result.to_csv(results_path + "_Consistency_Results"+dataset_tag+".csv")

    cm_exact = [[[0, 0, 0] for _ in range(2)] for _ in range(2)]
    cm_exec = [[[0, 0, 0] for _ in range(2)] for _ in range(2)]
    cm_exact_and_exec = [[[0, 0, 0] for _ in range(2)] for _ in range(2)]

    threshold_for_consistency = 0.5
    threshold_for_exact = 0.5
    threshold_for_execution = 0.5
    threshold_for_both = 0.5

    for index, row in result.iterrows():
        map = {}
        exact, exec, exact_and_exec, i, j = (0,)*5
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
            if m:
                exact += map[query]
            if e:
                exec += map[query]
            if m and e:
                exact_and_exec += map[query]

        querymax = max(map, key=lambda k: map[k])  # max frequency query
        
        # if more than the threshold, then they are consistent, hence you go to the second row
        # determine consistency
        if map[querymax] / repeat > threshold_for_consistency:
            i = 1
        else:
            i = 0

        exact = exact / repeat
        exec = exec / repeat
        exact_and_exec = exact_and_exec / repeat

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

        if exact_and_exec > threshold_for_both:
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
        plot_heatmap('Consistency on Exact Measure',df_cm_exact_numeric,df_cm_exact_labels,"_Exact_Measure")
        plot_heatmap('Consistency on Execution Measure',df_cm_exec_numeric,df_cm_exec_labels,"_Execution_Measure")
        plot_heatmap('Consistency on Both Exact and Execution Measure',df_cm_exact_and_exec_numeric,df_cm_exact_and_exec_labels,"_Exact_And_Execution_Measure")

    print('===========================================================================================')
    print("Finished Consistency Metric Evaluation")