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

def pick_hardness_lists(dataset, number_of_samples,repeat):
    with open(dataset, 'r') as f:
        data = json.load(f)
    
    simple = []
    moderate = []
    challenging = []

    for i, entry in enumerate(data):
        if entry.get("difficulty") == "simple":
            simple.append(i)
        elif entry.get("difficulty") == "moderate":
            moderate.append(i)
        elif entry.get("difficulty") == "challenging":
            challenging.append(i)

    # randomly sample indices from each category
    easy_sample = random.sample(simple, min(number_of_samples[0], len(simple)))
    medium_sample = random.sample(moderate, min(number_of_samples[1], len(moderate)))
    hard_sample = random.sample(challenging, min(number_of_samples[2], len(challenging)))
    
    repeated_easy = [item for item in easy_sample for _ in range(repeat)]
    repeated_medium = [item for item in medium_sample for _ in range(repeat)]
    repeated_hard = [item for item in hard_sample for _ in range(repeat)]

    combined_list = repeated_easy + repeated_medium + repeated_hard
    
    random.shuffle(combined_list)
    
    return combined_list

def get_hardness_index(hardness_level):
    if hardness_level == 'simple':
        return 0
    elif hardness_level == 'moderate':
        return 1
    elif hardness_level == 'challenging':
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
    
    number_of_samples = [1,1,1] #number of samples for each of the difficulty categories
    repeat = 1
    dataset = 'Datasets/bird/bird_dev_as_spider.json'
    # with_schema=False
    with_schema=True

    if with_schema: type = "with_schema/"
    else:    type = "without_schema/"
    
    # bird
    db_dir = "Datasets/bird/databases"
    table = "Datasets/bird/dev_tables.json"
    
    results_path = "Results/"
    model_path = 'Models/codegemma-7b-Q8_0.gguf'
    model_name = model_path.split('/')[-1].split('.gguf')[0]
    results_path= results_path + model_name

    file = open(dataset)
    data = json.load(file)
    
    final_list = pick_hardness_lists(dataset,number_of_samples,repeat)
    
    # result = pd.DataFrame(columns = ["Qid","Question","Hardness","Query Gold","Repeated","Query Generated"])
    result = pd.DataFrame(columns = ["Qid","Question","Db_Id","Hardness","Query Gold","Repeated","Query Generated"])

    for index in final_list:
        # schemaPath = "Datasets/spider/database/{}/{}.sqlite".format(data[index]['db_id'],data[index]['db_id'])
        schemaPath = "Datasets/bird/databases/{}/{}.sqlite".format(data[index]['db_id'],data[index]['db_id'])
        if with_schema: 
            schema = fetchSchema(schemaPath)
        else: schema = ""
        
        print("Schema:", schema)
        # print('\n\n',len(schema))
        
        response = model_predict(data[index]['question']+ schema)
        print("\nPredicted Output: ",response)
        
        if not result[result['Qid'] == index].empty:    # already exists
            row_index = result[result['Qid'] == index].index[0]
            result.loc[row_index,'Query Generated'] += "\n" + response

        else:
            curr = [index , data[index]['question'], data[index]['db_id'], data[index]['difficulty'],data[index]['query'],repeat,response]
            # curr = [index , data[index]['question'], data[index]['difficulty'],data[index]['query'],repeat,response]
            result.loc[len(result)] = curr
    result.to_csv(results_path + "_Consistency_Results_BIRD.csv")

    cm_exact = [[[0, 0, 0] for _ in range(2)] for _ in range(2)]
    cm_exec = [[[0, 0, 0] for _ in range(2)] for _ in range(2)]


    threshold_for_consistency = 0.5
    threshold_for_exact = 0.5
    threshold_for_execution = 0.5

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
            if m:
                exact += map[query]
            if e:
                exec += map[query]

        querymax = max(map, key=lambda k: map[k])  # max frequency query
        
        # if more than the threshold, then they are consistent, hence you go to the second row
        # determine consistency
        if map[querymax] / repeat > threshold_for_consistency:
            i = 1
        else:
            i = 0

        exact = exact / repeat
        exec = exec / repeat

        hardness_index = get_hardness_index(row['Hardness'])

        if exact > threshold_for_exact:
            j = 1
        else:
            j = 0
        cm_exact[i][j][hardness_index] += 1

        if exec > threshold_for_execution:
            j = 1
        else:
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

        #dataFrames with formatted labels
        df_cm_exact_labels = create_labeled_dataframe(cm_exact)
        df_cm_exec_labels = create_labeled_dataframe(cm_exec)

        # exact match
        plt.figure(figsize=(10, 6)) 
        plot1 = sns.heatmap(df_cm_exact_numeric, cmap="Blues", annot=df_cm_exact_labels, fmt="", annot_kws={"size": 12})
        plt.title('Consistency on Exact Measure') 
        plt.suptitle('n = ' + str(repeat), fontsize=10)
        plot1.figure.savefig(results_path + "_Exact_Measure_BIRD.png")  
        plt.close()

        # execution accuracy
        plt.figure(figsize=(10, 6))  
        plot2 = sns.heatmap(df_cm_exec_numeric, cmap="Blues", annot=df_cm_exec_labels, fmt="", annot_kws={"size": 12})
        plt.title('Consistency on Execution Measure') 
        plt.suptitle('n = ' + str(repeat), fontsize=10)
        plot2.figure.savefig(results_path + "_Execution_Measure_BIRD.png")  
        plt.close()

    print('===========================================================================================')
    print("Finished Evaluation")
    
    # CHEKING FOR CONSISTENCY IN ANSWERS  
    #
    #   1  |  2         
    #   3  |  4         
    # 1 - Queries are inconsistent and != GOLD query    2 - Queries are inconsistent but == GOLD query
    # 3 - Queries are consistent but != GOLD query.     4 - Queries are consistent and == GOLD query
    # consistent query : more than half the time returns the SAME query
    # == GOLD query - 3 types of equality : Exact match, Execution accuracy, Partial Matching
