import os, sqlite3

def execute_sql(predicted_sql, ground_truth_sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        print('----------------------------------------------------------')
        print(set(predicted_res))
        print('----------------------------------------------------------')
        
        cursor.execute(ground_truth_sql)
        ground_truth_res = cursor.fetchall()
        print('----------------------------------------------------------')
        print(set(ground_truth_res))
        print('----------------------------------------------------------')
        
        res = 1 if predicted_res == ground_truth_res else 0
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        res = 0  # Return 0 in case of an error
    finally:
        conn.close() 
    return res

def execute_sql_sym_diff(predicted_sql, ground_truth_sql, db_path):
    if predicted_sql == 'NULL':
        print("Predicted SQL is None")
        return 0
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        sym_diff_sql = f"""
        SELECT * FROM ({predicted_sql})
        EXCEPT
        SELECT * FROM ({ground_truth_sql})
        UNION
        SELECT * FROM ({ground_truth_sql})
        EXCEPT
        SELECT * FROM ({predicted_sql})
        """
        cursor.execute(sym_diff_sql)
        sym_diff_res = cursor.fetchall()
        res = 1 if not sym_diff_res else 0 
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        res = 0  
    finally:
        conn.close() 
    return res

gold = "SELECT count(*) FROM concert WHERE YEAR  =  2014 OR YEAR  =  2015"
pred = "SELECT count(*) FROM concert WHERE YEAR  IN (2014, 2015)"

path = os.path.join("Datasets/spider/database", 'concert_singer', 'concert_singer' + ".sqlite")

sym_result = execute_sql_sym_diff(pred, gold, path)

result=execute_sql(pred, gold, path)

print("Symmetric difference: Are queries equivalent?", bool(sym_result))
print("Normal: Are queries equivalent?", bool(result))

