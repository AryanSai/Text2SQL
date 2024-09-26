import argparse, torch,json,pandas as pd,seaborn as sns, matplotlib.pyplot as plt
import os, sqlite3

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
    plot.figure.savefig(results_path + measure_name +model_name+".png")  
    plt.close()

def execute_sql(predicted_sql, ground_truth_sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
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

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(0)
        free_memory =  gpu_properties.total_memory - torch.cuda.memory_allocated(0)
    else:
        free_memory =  0
        
    print('===========================================================================================')
    print(f"Free GPU Memory: {free_memory / (1024 ** 3):.2f} GB")
    print('===========================================================================================')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, required=True)
    parser.add_argument('--dbdir', dest='dbdir', type=str, required=True)
    parser.add_argument('--dbtable', dest='dbtable', type=str, required=True)
    parser.add_argument('--inputjson', dest='inputjson', type=str, required=True)
    parser.add_argument('--output', dest='output', type=str, required=True)
    parser.add_argument('--exact_threshold', dest='exact_threshold', type=float, required=True)
    parser.add_argument('--exec_threshold', dest='exec_threshold', type=float, required=True)
    parser.add_argument('--consistency_threshold', dest='consistency_threshold', type=float, required=True)
    parser.add_argument('--repeat', dest='repeat', type=int, required=True)
    
    args = parser.parse_args()

    model_name=args.model
    db_dir = args.dbdir
    table = args.dbtable
    input_json = args.inputjson
    results_path = args.output
    threshold_for_execution = args.exec_threshold
    threshold_for_exact = args.exact_threshold
    threshold_for_consistency = args.consistency_threshold
    repeat = args.repeat
    
    results_path = results_path + model_name

    cm_exec = [[[0, 0, 0] for _ in range(2)] for _ in range(2)]
    
    with open(input_json, 'r') as file:
        result = json.load(file)

    for entry in result:
        map = {}
        exec, i, j = (0,)*3
        
        # predicted_queries = entry['pred']# multiple predictions
        predicted_queries = entry.get('pred', [])
        
        # counting the frequencies of the predictions
        for r in predicted_queries:  # result will look like : {'SELECT * FROM tablename': 5}
            if r in map:
                map[r] += 1
            else:
                map[r] = 1

        for query in map:
            db=entry['db_id']
            pred = query
            db = os.path.join(db_dir, db, db + ".sqlite")
            e = execute_sql(pred,entry['gold'] ,db)
            print(f"Query: {pred}\n Execution: {e}")
            if e:
                exec += map[query]

        querymax = max(map, key=lambda k: map[k])  # max frequency query
        
        # if more than the threshold, then they are consistent, hence you go to the second entry
        # determine consistency
        if map[querymax] / repeat > threshold_for_consistency:
            i = 1
        else:
            i = 0

        exec = exec / repeat

        difficulty_index = get_difficulty_index(entry['difficulty'])

        if exec > threshold_for_execution:
            j = 1
        else:
            j = 0
        cm_exec[i][j][difficulty_index] += 1

        df_cm_exec_numeric = pd.DataFrame(
            [[sum(cm_exec[i][j]) for j in range(2)] for i in range(2)],
            columns=["0", "1"],
            index=["0", "1"]
        )
        
        #dataFrames with formatted labels
        df_cm_exec_labels = create_labeled_dataframe(cm_exec)

        # plot the heatmaps
        plot_heatmap('Consistency on Execution Measure',df_cm_exec_numeric,df_cm_exec_labels,"_Execution_Measure")
    
    print('===========================================================================================')
    print("Finished Consistency Metric Evaluation")