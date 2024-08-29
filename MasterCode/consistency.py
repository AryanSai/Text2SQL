import seaborn as sns
import matplotlib.pyplot as plt
import re,json,random,pandas as pd,sqlite3,os,sys
from llama_cpp import Llama
from test_eval import main

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
    Get database's schema, which is a dict with table name as key
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
    for key in schema:
        st += key + " : " + str(schema[key]) + "\n"
    return st

def model_predict(question):
    template = "Question: Convert the following text to an SQLite query and end the query with a semi-colon(;). Please provide only the query without any explanation: " 
    with suppress_stdout_stderr():
        llm = Llama(model_path = model_path)
        output = llm(
            prompt = template + question + "\nAnswer:",
            max_tokens=300,
            temperature=0.1,
            stop=["This ","Explanation: ","Note:","The",";"],
            echo=True
        )
            
    response = output['choices'][0]['text']
    # print("\nOutput After: ", output)
    match = re.search("SELECT .*",response,re.DOTALL | re.IGNORECASE)
    str = ""
    if(match):  
        str = match.group(0)
    else:  str = "NULL"
    str = str.replace(";","")
    str = str.replace("```","")
    str = str.replace("\n",' ')
    # print("\nOutput After after: ", str)
    return str

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--size', dest='size', type=int)  #No of questions
    # parser.add_argument('--repeat', dest='repeat', type=int)
    # parser.add_argument('--test', dest='test', type=str)
    # parser.add_argument('--model', dest='model', type=str)
    # parser.add_argument('--schema',dest='schema',type=int)
    # args = parser.parse_args()

    # questionSize = args.size
    # repeat = args.repeat
    # dataset = args.test
    # model_name = args.model
    # schm=args.schema
    
    questionSize = 5
    repeat = 5
    dataset = 'Datasets/spider/dev.json'
    model_name = 'CodeGemma7B_WithHardness_'
    # schm=False
    schm=True

    if schm: type = "with_schema/"
    else:    type = "without_schema/"
    
    results_path = "Results/"
    model_path = 'Models/codegemma-7b-Q8_0.gguf'
    results_path= results_path + model_name.split(".")[0]

    f = open(dataset)
    data = json.load(f)

# creating list of questions to be taken from the file. the size of the list is 'questionSize' from arguments
    lst = []
    r = random.randrange(0,len(data),1)
    for i in range(0,questionSize):
        # print("\n",i)
        while r in lst:
            r = random.randrange(0,len(data),1)
        lst.append(r)
    newlist = [item for item in lst for i in range(repeat)]
    
    random.shuffle(newlist)
    # print(newlist)

    result = pd.DataFrame(columns = ["Qid","Question","Query Gold","Repeated","Query Generated"])

    for index in newlist:
        schemaPath = "Datasets/spider/database/{}/{}.sqlite".format(data[index]['db_id'],data[index]['db_id'])
        # print(schemaPath)
        if schm: 
            schema = fetchSchema(schemaPath)
        else: schema = ""
        
        resp = model_predict(data[index]['question']+ schema)
        print("\nPredicted Output: ",resp)
        
        if not result[result['Qid'] == index].empty:    # already exists
            row_index = result[result['Qid'] == index].index[0]
            # result.loc[row_index,'repeated']+=1
            result.loc[row_index,'Query Generated'] += "\n" + resp

        else:
            curr = [index , data[index]['question'] , data[index]['query'],repeat,resp]
            result.loc[len(result)] = curr
    result.to_csv(results_path + "consistency_results.csv")

    cm_exact = [[0,0],[0,0]]
    cm_exec = [[0,0],[0,0]]

    threshold_for_consistency = 0.5
    threshold_for_exact = 0.5
    threshold_for_execution = 0.5

    for index,row in result.iterrows():
        map = {}
        exact,exec,i,j = (0,)*4
        qid = row['Qid']
        lst = row['Query Generated'].split("\n")
        
        for r in lst:    # result will look like : {'SELECT * FROM tablename': 5}
            if r in map:
                map[r]+=1
            else: map[r] = 1
        for key in map:
            gold = row['Query Gold'] + "\t" + data[qid]['db_id']
            pred = key
            print(gold,key)
            m,e = main(gold,pred)    #returns the exact match and execution accuracy
            if m: exact+=map[key]
            if e : exec+=map[key]

        keymax = max(map,key = lambda k:map[k])
        if map[keymax]/repeat > threshold_for_consistency:
            i=1
            print(index,keymax,map[keymax])
        else: i=0

        exact = exact/repeat
        exec = exec/repeat    #repeats_per_question
        
        if exact>threshold_for_exact: j=1
        else: j=0
        cm_exact[i][j]+=1

        if exec>threshold_for_execution: j=1
        else: j=0
        cm_exec[i][j]+=1

    cm = pd.DataFrame(cm_exact)
    plt.figure(figsize=(5,3))
    plot1 = sns.heatmap(cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
    plot1.figure.savefig(results_path + "cm_exact")
    # plt.show()
    
    cm = pd.DataFrame(cm_exec)
    plt.figure(figsize=(5,3))
    plot2 = sns.heatmap(cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
    plot2.figure.savefig(results_path + "cm_exec")
    # plt.show()
    

    # CHEKING FOR CONSISTENCY IN ANSWERS  

    #
    #   1  |  2         
    #   3  |  4         
    #                   1 - Queries are inconsistent and != GOLD query    2 - Queries are inconsistent but == GOLD query
    #                   3 - Queries are consistent but != GOLD query.     4 - Queries are consistent and == GOLD query
    # consistent query : more than half the time returns the SAME query
    # == GOLD query - 3 types of equality : Exact match, Execution accuracy, Partial Matching
