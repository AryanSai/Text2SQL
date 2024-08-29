import argparse,re,sys, numpy as np,time,math,sqlite3, multiprocessing as mp, csv
from func_timeout import func_timeout, FunctionTimedOut
import matplotlib.pyplot as plt,numpy as np

def result_callback(result):
    exec_result.append(result)

def clean_abnormal(input):
    input = np.asarray(input)
    processed_list = []
    mean = np.mean(input,axis=0)
    std = np.std(input,axis=0)
    for x in input:
        if x < mean + 3 * std and x > mean - 3 * std:
            processed_list.append(x)
    return processed_list

def execute_sql(sql, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    start_time = time.time()
    cursor.execute(sql)
    exec_time = time.time() - start_time
    return exec_time

def iterated_execute_sql(predicted_sql,ground_truth,db_path,iterate_num):
    conn = sqlite3.connect(db_path)
    diff_list = []
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    time_ratio = 0
    # print(predicted_res,ground_truth_res)
    if set(predicted_res) == set(ground_truth_res):
        for i in range(iterate_num):
            predicted_time = execute_sql(predicted_sql, db_path)
            ground_truth_time = execute_sql(ground_truth, db_path)
            diff_list.append(ground_truth_time / predicted_time)
        processed_diff_list = clean_abnormal(diff_list)
        time_ratio = sum(processed_diff_list) / len(processed_diff_list)
    return time_ratio

def execute_model(predicted_sql,ground_truth, db_place, idx, iterate_num, meta_time_out):
    try:
        # you can personalize the total timeout number
        # larger timeout leads to more stable ves
        # while it needs more your patience....
        time_ratio = func_timeout(meta_time_out * iterate_num, iterated_execute_sql,
                                  args=(predicted_sql, ground_truth, db_place, iterate_num))
        # print([idx, math.sqrt(time_ratio)])
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f'timeout',)]
        time_ratio = 0
    except Exception as e:
        result = [(f'error',)]  # possibly len(query) > 512 or not executable
        time_ratio = 0
    result = {'sql_idx': idx, 'time_ratio': time_ratio}
    return result

def run_sqls_parallel(num_iterations, sqls, db_ids, num_cpus=1, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i,sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        db_path = "Datasets/spider/database/{}/{}.sqlite".format(db_ids[i],db_ids[i])
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_path , i, num_iterations, meta_time_out), callback=result_callback)
    pool.close()
    pool.join()

def sort_results(list_of_dicts):
  return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_ves(exec_results):
    num_queries = len(exec_results)
    
    if(num_queries==0):
        return 0

    total_ratio = 0
    count = 0

    for i, result in enumerate(exec_results):
        if result['time_ratio'] != 0:
            count += 1
        total_ratio += math.sqrt(result['time_ratio']) * 100
    ves = (total_ratio/num_queries)
    return ves


def compute_ves_by_diff(exec_results,challengingness_list):
    num_queries = len(exec_results)
    simple_results, moderate_results, challenging_results = [], [], []
    for i,challengingness in enumerate(challengingness_list):
        if challengingness == 'simple':
            simple_results.append(exec_results[i])
        if challengingness == 'moderate':
            moderate_results.append(exec_results[i])
        if challengingness == 'challenging':
            challenging_results.append(exec_results[i])
    simple_ves = compute_ves(simple_results)
    moderate_ves = compute_ves(moderate_results)
    challenging_ves = compute_ves(challenging_results)
    all_ves = compute_ves(exec_results)
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_ves, moderate_ves, challenging_ves, all_ves, count_lists

def print_data(score_lists,count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))

    print('========================================   VES   ========================================')
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('ves', *score_lists))


def plot_ves(score_lists, model_name, num_iterations):
    save_path=model_name+'_VES_Score.png'
    levels = ['simple', 'moderate', 'challenging', 'total']
    max_score = max(score_lists) if score_lists else 1  
    normalized_scores = [score / max_score for score in score_lists]
    colors = [plt.cm.Blues(score) for score in normalized_scores]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(levels, score_lists, color=colors)
    plt.xlabel('Difficulty Level')
    plt.ylabel('VES Score')
    plt.title(model_name + ' - VES Scores')
    plt.suptitle('Iterations = ' + str(num_iterations))
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 2), ha='center', va='bottom')
    plt.savefig(save_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', dest='input_file', type=str, required=True)
    parser.add_argument('--iterations', dest='iterations', type=int, default=20)
    
    args = parser.parse_args()
    
    input_file = args.input_file
    num_iterations = args.iterations
    
    gold_queries=[]
    pred_queries=[]
    db_ids=[]
    challengingness_list=[]
    
    with open(input_file, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        
        idx_query_gold = header.index("Query Gold")
        idx_query_generated = header.index("Most Frequent Query")
        idx_db_id = header.index("Db_Id")
        idx_challengingness = header.index("Difficulty")

        for row in csv_reader:
            gold_queries.append(row[idx_query_gold])
            pred_queries.append(row[idx_query_generated])
            db_ids.append(row[idx_db_id])
            challengingness_list.append(row[idx_challengingness])

    exec_result = []
    
    query_pairs = list(zip(pred_queries, gold_queries))
    
    run_sqls_parallel(num_iterations, query_pairs, db_ids, num_cpus=1, meta_time_out=30.0)
    exec_result = sort_results(exec_result)
    print('Started Evaluation')
    
    simple_ves, moderate_ves, challenging_ves, ves, count_lists = compute_ves_by_diff(exec_result, challengingness_list)
    score_lists = [simple_ves, moderate_ves, challenging_ves, ves]
    
    print_data(score_lists, count_lists)
    print('===========================================================================================')
    print("Finished VES evaluation")
    
    match = re.search(r'([^/]+)_Consistency_Results.*\.csv$', input_file)
    if match:
        model_name = match.group(1)
    else:
        model_name = None
    
    plot_ves(score_lists,model_name, num_iterations)