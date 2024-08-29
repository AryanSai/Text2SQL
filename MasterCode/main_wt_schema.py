from llama_cpp import Llama
import json, re
import pandas as pd

model_name = "Meta-Llama-3.1-8B-Instruct-IQ3_M"
csv_file_name = "Results/"  + model_name + "_with_schema_new.csv"
llm = Llama(model_path="Models/Meta-Llama-3.1-8B-Instruct-IQ3_M.gguf",n_ctx=2048, n_gpu_layers=-1)
print("model loaded")

f = open("Datasets/spider/dev.json")
data = json.load(f)

# extract tablename with columns and its foreign keys
def fetchSchema(schemapath):
    final = ""
    fkeys = "["
    with open(schemapath, "r", encoding="utf-8") as line:
        # print("Line:",line)
        text = line.read()
    
    lines = text.split("\n")
    # print("Lines:",lines)
    i=1
    for i in range(len(lines)):
        if(lines[i].startswith("CREATE")):
            tablename = lines[i].split(" ")[2].strip("\"")
            final = final + "TABLE {}, columns = [*".format(tablename)
            i+=1
            while(not(lines[i].startswith("PRIMARY"))):
                final = final +","+ lines[i].split(" ")[0].strip("\"")
                i+=1
            final = final+"]\n"
        if(lines[i].startswith("FOREIGN")):
            tmp = lines[i].split(" ")
            l1 = tmp[2].strip("()\"")
            l2 = tmp[4].split("(")[0].strip("\"`")
            # print(l1,l2)
            fkeys+= "{}.{} = {}.{},".format(tablename,l1,l2,l1)
    fkeys+="]"
    final+="Foreign Keys = " + fkeys 
    return final


df = pd.DataFrame(columns = ["Question","Query Gold","Query Generated"])

template = "Question: Convert the following text to sqlite query: " 
i=1
for item in data[:6]:
    print(i)
    try:
        temp = [item['question'],item['query']]
        schemaPath = "Datasets/spider/database/{}/schema.sql".format(item['db_id'])
        schema = fetchSchema(schemaPath)
        # print(schema)
        output = llm(
            # "Question: Convert the following text statement to sqlite query : display all names of students whose marks are above average of all marks of students. Answer:",
            prompt= template + item['question'] + "\nThe schema is given by : " + schema + "\nAnswer:",
            max_tokens=300,
            temperature=0.1,
            stop=["Explanation:"],  # for llama2-7b
            echo=True
        )
        # print(json.dumps(output,indent=2))
        response = output["choices"][0]["text"]
#        print(response + "\n")
        match = re.search("SELECT .*",response,re.DOTALL)
        if(match):  
            print("match")
            temp.append(match.group(0))
        else:   temp.append("NULL")
        df.loc[len(df)] = temp
    except Exception as error:
        i+=1
        print(error)
        continue
    i+=1
df.to_csv(csv_file_name)