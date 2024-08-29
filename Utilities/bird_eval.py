import sqlite3

def execute_sql(predicted_sql,ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res

def main():
    gold = 'SELECT name FROM singer WHERE birth_year IN (1948, 1949)'
    pred = 'SELECT Name FROM singer WHERE Birth_Year = 1948 OR Birth_Year = 1949'
    db_path = 'Datasets/spider/database/singer/singer.sqlite'
    print(execute_sql(pred,gold,db_path))
    
main()