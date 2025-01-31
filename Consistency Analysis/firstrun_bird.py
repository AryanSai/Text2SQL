import argparse
import sqlite3, pandas as pd, os

def execute_sql(predicted_sql, ground_truth_sql, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth_sql)
        ground_truth_res = cursor.fetchall()
        res = 1 if set(predicted_res) == set(ground_truth_res) else 0
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        res = 0  # Return 0 in case of an error
    finally:
        conn.close() 
    return res

def process_csv(file_path):
    db_dir="Datasets/bird/databases"
    df = pd.read_csv(file_path)
    execution_matches = 0
    total_queries = 0
    
    difficulty_levels = ['simple', 'moderate', 'challenging']
    exec_by_hardness = {level: 0 for level in difficulty_levels}
    total_by_hardness = {level: 0 for level in difficulty_levels}
    
    for idx, row in df.iterrows():
        generated_queries = row['Query Generated'].split('\n')  
        first_prediction = generated_queries[0]
        
        hardness = row['Difficulty']
        db_id = row['Db_Id']
        db_path = os.path.join(db_dir, db_id, db_id + ".sqlite")
        exec_match = execute_sql(first_prediction, row['Query Gold'], db_path)
        df.loc[idx, 'Execution Accuracy'] = exec_match
        
        if exec_match:
            execution_matches += 1
            exec_by_hardness[hardness] += 1
        
        total_queries += 1
        total_by_hardness[hardness] += 1
        
    new_file_path = os.path.splitext(file_path)[0] + '_with_scores.csv'
    df.to_csv(new_file_path, index=False)
    
    print(f"Total Queries Processed: {total_queries}")
    print(f"Execution Matches: {execution_matches}")
    
    print("\nMatches by Hardness Level:")
    for level in difficulty_levels:
        print(f"{level.capitalize()} - Exec: {exec_by_hardness[level]}, Total: {total_by_hardness[level]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', dest='input_file', type=str, required=True)
    
    args = parser.parse_args()
    
    csv_file = args.input_file
    
    process_csv(csv_file)
