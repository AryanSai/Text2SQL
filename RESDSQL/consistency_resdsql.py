import argparse, torch,json,pandas as pd,seaborn as sns, matplotlib.pyplot as plt
from evaluation import main

def get_hardness_index(hardness_level):
    if hardness_level == 'easy':
        return 0
    elif hardness_level == 'medium':
        return 1
    elif hardness_level == 'hard':
        return 2
    elif hardness_level == 'extra':
        return 3
    else:
        raise ValueError("Unknown hardness level")

def create_labeled_dataframe(matrix):
    # """Creates a DataFrame with formatted labels for each cell."""
    data = [[format_cell_value(matrix, i, j) for j in range(2)] for i in range(2)]
    return pd.DataFrame(data, columns=["0", "1"], index=["0", "1"])

def format_cell_value(matrix, i, j):
    values = matrix[i][j]
    return (f"Overall: {sum(values)}\nEasy: {values[0]}\nMedium: {values[1]}"
            f"\nHard: {values[2]}\nExtra Hard: {values[3]}")

def plot_heatmap(title,df,labels,measure_name):
    plt.figure(figsize=(10, 6)) 
    plot = sns.heatmap(df, cmap="Blues", annot=labels, fmt="", annot_kws={"size": 12})
    plt.title(title) 
    plt.suptitle('n = ' + str(repeat), fontsize=10)
    plot.figure.savefig(results_path + measure_name +model_name+".png")  
    plt.close()

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

    cm_exact = [[[0, 0, 0,0] for _ in range(2)] for _ in range(2)]
    cm_exec = [[[0, 0, 0,0] for _ in range(2)] for _ in range(2)]
    cm_exact_and_exec = [[[0, 0, 0,0] for _ in range(2)] for _ in range(2)]
    
    with open(input_json, 'r') as file:
        result = json.load(file)

    for entry in result:
        map = {}
        exact, exec, i, j = (0,)*4
        
        # predicted_queries = entry['pred']# multiple predictions
        predicted_queries = entry.get('pred', [])
        
        # counting the frequencies of the predictions
        for r in predicted_queries:  # result will look like : {'SELECT * FROM tablename': 5}
            if r in map:
                map[r] += 1
            else:
                map[r] = 1

        for query in map:
            gold = entry['gold'] + "\t" + entry['db_id']
            pred = query
            print('=============================')
            print('gold= ',gold)
            print('pred= ',pred)
            print('=============================')
            m, e = main(gold, pred, db_dir, table)  # returns the exact match and execution accuracy
            
            print(f"Query: {pred}\nExact Match: {m}, Execution: {e}")
            
            if m:
                exact += map[query]
            if e:
                exec += map[query]

        querymax = max(map, key=lambda k: map[k])  # max frequency query
        
        # if more than the threshold, then they are consistent, hence you go to the second entry
        # determine consistency
        if map[querymax] / repeat > threshold_for_consistency:
            i = 1
        else:
            i = 0

        exact = exact / repeat
        exec = exec / repeat

        difficulty_index = get_hardness_index(entry['hardness'])

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

        if exec > threshold_for_execution and exact > threshold_for_exact:
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