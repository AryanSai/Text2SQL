import json, os
from spidermaster.evaluation import Evaluator
from spidermaster.process_sql import Schema, get_schema, get_sql
    
f = open("Datasets/spider/dev.json")
data = json.load(f)

for item in data: 
    db = os.path.join('Datasets/spider/database', item['db_id'], item['db_id'] + ".sqlite")
    schema = Schema(get_schema(db))
    g_sql = get_sql(schema, item['query'])
    evaluator = Evaluator()
    
    item['hardness'] = evaluator.eval_hardness(g_sql)

output_file = "dev_with_hardness.json"
with open(output_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"New JSON file created with hardness added: {output_file}")
