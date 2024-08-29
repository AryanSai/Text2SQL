import re,json,random,pandas as pd,sqlite3,os,sys
from llama_cpp import Llama

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

def model_predict(question,model_path):
    template = "Question: Convert the following text to an SQLite query and end the query with a semi-colon(;). Please provide only the query without any explanation: " 
    with suppress_stdout_stderr():
        llm = Llama(model_path = model_path)
        output = llm(
            prompt = template + question + "\nAnswer:",
            max_tokens=300,
            temperature=0.1,
            stop=["This ", "Explanation: ", "Note:", "The", ";", r"\\", r"\end{code}", "Comment:"],
            echo=True
        )
            
    responseonse = output['choices'][0]['text']
    match = re.search("SELECT .*",responseonse,re.DOTALL | re.IGNORECASE)
    str = ""
    if(match):  
        str = match.group(0)
    else:  str = "NULL"
    str = str.replace(";","")
    str = str.replace("```","")
    str = str.replace("\n",' ')
    return str

def pick_hardness_lists(dataset, number_of_samples,repeat):
    with open(dataset, 'r') as f:
        data = json.load(f)
    
    easy = []
    medium = []
    hard = []

    for i, entry in enumerate(data):
        if entry.get("hardness") == "easy":
            easy.append(i)
        elif entry.get("hardness") == "medium":
            medium.append(i)
        elif entry.get("hardness") == "hard":
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

def get_hardness_index(hardness_level):
    if hardness_level == 'easy':
        return 0
    elif hardness_level == 'medium':
        return 1
    elif hardness_level == 'hard':
        return 2
    else:
        raise ValueError("Unknown hardness level")

def main(number_of_samples=[1,1,1],repeat=1,with_schema=True):
    dataset = 'Datasets/spider_dev_with_hardness.json'

    results_path = "Predictions/"
    model_path = 'Models/codegemma-7b-Q8_0.gguf'
    model_name = model_path.split('/')[-1].split('.gguf')[0]
    results_path= results_path + model_name

    file = open(dataset)
    data = json.load(file)
    
    final_list = pick_hardness_lists(dataset,number_of_samples,repeat)
    
    result = pd.DataFrame(columns = ["Qid","Question","Db_Id","Hardness","Query Gold","Repeated","Query Generated"])

    for index in final_list:
        schemaPath = "Datasets/spider/database/{}/{}.sqlite".format(data[index]['db_id'],data[index]['db_id'])
        if with_schema: 
            schema = fetchSchema(schemaPath)
        else: schema = ""
        
        response = model_predict(data[index]['question']+ schema,model_path)
        print("\nPredicted Output: ",response)
        
        if not result[result['Qid'] == index].empty:    # already exists
            row_index = result[result['Qid'] == index].index[0]
            result.loc[row_index,'Query Generated'] += "\n" + response

        else:
            curr = [index , data[index]['question'], data[index]['db_id'], data[index]['hardness'],data[index]['query'],repeat,response]
            result.loc[len(result)] = curr
    result.to_csv(results_path + "_Predicted_Results.csv")

    print('===========================================================================================')
    print("Finished Prediction")

main()