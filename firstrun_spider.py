import pandas as pd, os
from evaluation import main  

def process_csv(file_path, db_dir,table_dir):
    df = pd.read_csv(file_path)
    
    exact_matches = 0
    execution_matches = 0
    total_queries = 0
    
    difficulty_levels = ['easy', 'medium', 'hard', 'extra']
    exact_by_hardness = {level: 0 for level in difficulty_levels}
    exec_by_hardness = {level: 0 for level in difficulty_levels}
    total_by_hardness = {level: 0 for level in difficulty_levels}
    
    for idx, row in df.iterrows():
        qid = row['Qid']
        gold_query = row['Query Gold']
        generated_queries = row['Query Generated'].split('\n')  
        first_prediction = generated_queries[0]
        
        hardness = row['hardness']
        db_id = row['Db_Id']
        gold = gold_query + "\t" + db_id
        
        # db_path = db_dir+"{}/{}.sqlite".format(db_id,db_id)
        exact_match, exec_match = main(gold, first_prediction,  db_dir,table_dir)
        
        df.loc[idx, 'Exact Match'] = exact_match
        df.loc[idx, 'Execution Accuracy'] = exec_match
        
        if exact_match:
            exact_matches += 1
            exact_by_hardness[hardness] += 1
        
        if exec_match:
            execution_matches += 1
            exec_by_hardness[hardness] += 1
        
        total_queries += 1
        total_by_hardness[hardness] += 1
        
    new_file_path = os.path.splitext(file_path)[0] + '_with_scores.csv'
    df.to_csv(new_file_path, index=False)
    
    print(f"Total Queries Processed: {total_queries}")
    print(f"Exact Matches: {exact_matches}")
    print(f"Execution Matches: {execution_matches}")
    
    print("\nMatches by Hardness Level:")
    for level in difficulty_levels:
        print(f"{level.capitalize()} - Exact: {exact_by_hardness[level]}, Exec: {exec_by_hardness[level]}, Total: {total_by_hardness[level]}")
    
    return exact_by_hardness, exec_by_hardness, total_by_hardness

csv_file = 'Results/Spider/1000samples/deepseek/DeepSeek-Coder-V2-Lite-2.4B-Instruct-Q8_0DeepSeek-Coder-V2-Lite-2.4B-Instruct-Q8_0_Consistency_Results_SPIDER.csv'  # path to your CSV file
new_file_path = "Results/Spider/"
db_directory="Datasets/spider/database"
table_dir="Datasets/spider/tables.json"
exact_scores, exec_scores, total_by_level = process_csv(csv_file, db_directory,table_dir)
