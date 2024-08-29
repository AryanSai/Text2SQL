import seaborn as sns
import matplotlib.pyplot as plt
import re
import json
import random
import argparse
import pandas as pd
#import google.generativeai as palm
import requests
import sqlite3
import os,sys
from llama_cpp import Llama
from test_eval import main

#palm.configure(api_key='AIzaSyCtgDN2xBVB2fLINckHC94V6eqmt3KcijU')

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
    # print(st)
    return st

def model1_predict(question):   #PALM
    response = "Question: Convert the following text statement to sqlite query: " 
 #   output = palm.generate_text(prompt= (template + question))
 #   response = output.result
    # print(response)
    match = re.search("SELECT .*",response,re.DOTALL | re.IGNORECASE)
    str = ""
    if(match):  
        str = match.group(0)
        # print("match\n",str)
    else:  str = "NULL"
    str = str.replace(";","")
    str = str.replace("```","")
    str = str.replace("\n",' ')
    return str

def model2_predict(question):
    template = "Question: Convert the following text to sqlite query: " 
    prompt = template + question + "\nAnswer: "
    with suppress_stdout_stderr():
        llm = Llama(model_path = model_path)
        output = llm(
            prompt= template + question + "\nAnswer:",
            max_tokens=300,
            temperature=0.1,
            stop=["Explanation:"],  # for llama2-7b
            echo=False
        )
        response = output['choices'][0]['text']
    match = re.search("SELECT .*",response,re.DOTALL | re.IGNORECASE)
    str = ""
    if(match):  
        str = match.group(0)
        # print("match\n",str)
    else:  str = "NULL"
    str = str.replace(";","")
    str = str.replace("```","")
    str = str.replace("\n",' ')
    return str

def query(payload):
    endpoint = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
    header = {"Authorization" : f"Bearer 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"}
    response = requests.request("POST",endpoint,headers=header,json=payload)
    # print(response)
    return json.loads(response.content.decode("utf-8"))

def model3_predict(question):
    template = "Q: Convert the following text to SQL query: "
    input_text = template + question + "A:"
    resp = query({"inputs" : input_text, "options" : {"wait_for_model" : True,"max_length" : 100}})
    # print(resp)
    response = resp[0]['generated_text']
    print(response)
    match = re.search("SELECT .*",response,re.DOTALL | re.IGNORECASE)
    str = ""
    if(match):  
        str = match.group(0)
        # print("match\n",str)
    else:  str = "NULL"
    str = str.replace(";","")
    str = str.replace("```","")
    str = str.replace("\n",' ')
    return str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', dest='size', type=int)  #No of questions
    parser.add_argument('--repeat', dest='repeat', type=int)
    parser.add_argument('--test', dest='test', type=str)
    parser.add_argument('--model', dest='model', type=str)
    parser.add_argument('--schema',dest='schema',type=int)
    args = parser.parse_args()

    questionSize = args.size
    repeat = args.repeat
    dataset = args.test
    model_name = args.model
    schm=args.schema

    model_path = "models/"
    results_path = "results/"

    if schm: type = "with_schema/"
    else:    type = "without_schema/"
    model_path=model_path + model_name
    results_path= results_path + model_name.split(".")[0] + "/" + type   #removes the gguf extension.       

    f = open(dataset)
    data = json.load(f)

# creating list of questions to be taken from the file. the size of the list is 'questionSize' from arguments
    lst = []
    r = random.randrange(0,len(data),1)
    for i in range(0,questionSize):
        while r in lst:
            r = random.randrange(0,len(data),1)
        lst.append(r)
    print(lst)
    newlist = [item for item in lst for i in range(repeat)]
    
    random.shuffle(newlist)
    # print(newlist)

    result = pd.DataFrame(columns = ["Qid","Question","Query Gold","repeated","Query Generated"])

    for index in newlist:
        schemaPath = "./spider/database/{}/{}.sqlite".format(data[index]['db_id'],data[index]['db_id'])
        print(schemaPath)
        if schm: 
            schema = fetchSchema(schemaPath)
            # schema = "The schema is given by: " + schema
        else: schema = ""
        
        if(model_name == 'Palm'):
            resp = model1_predict(data[index]['question']+ schema)
        elif(model_name=="Flan-t5"):
            resp = model3_predict(data[index]['question']+ schema)
        else : resp = model2_predict(data[index]['question']+ schema)
        print(resp)
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
            # pred = 'SELECT T1.Airline FROM AIRLINES AS T1 JOIN FLIGHTS AS T2 ON T1.uid  =  T2.Airline GROUP BY T1.Airline HAVING count(*)  <  200'
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
    plot1.figure.savefig(results_path + "cm_exec")
    # plt.show()
    

    # CHEKING FOR CONSISTENCY IN ANSWERS  

    #
    #   1  |  2         
    #   3  |  4         
    #                   1 - Queries are inconsistent and != GOLD query    2 - Queries are inconsistent but == GOLD query
    #                   3 - Queries are consistent but != GOLD query.     4 - Queries are consistent and == GOLD query
    # consistent query : more than half the time returns the SAME query
    # == GOLD query - 3 types of equality : Exact match, Execution accuracy, Partial Matching
