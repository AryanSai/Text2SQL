import pandas as pd
import sqlite3, os

def execute_symmetric_difference(gold_sql, with_desc_sql, db_path):
    """Executes the symmetric difference between Gold and With Description SQL queries."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT * FROM ({gold_sql}) LIMIT 1")
        gold_cols = len(cursor.description)
        
        cursor.execute(f"SELECT * FROM ({with_desc_sql}) LIMIT 1")
        with_desc_cols = len(cursor.description)

        if gold_cols != with_desc_cols:
            print(f"Column mismatch: Gold({gold_cols}) vs With Description({with_desc_cols})")
            return 0  
        
        sym_diff_sql = f"""
        SELECT * FROM ({gold_sql}) AS gold
        EXCEPT
        SELECT * FROM ({with_desc_sql}) AS with_desc
        UNION
        SELECT * FROM ({with_desc_sql}) AS with_desc
        EXCEPT
        SELECT * FROM ({gold_sql}) AS gold
        """

        cursor.execute(sym_diff_sql)
        sym_diff_res = cursor.fetchall()

        return 1 if not sym_diff_res else 0  

    except sqlite3.Error as e:
        print(f"SQL Execution Error: {e}")
        return 0

    finally:
        conn.close()

csv_path = "Description-based Analysis/SIngleRun/yicoder_desc_analysis_results.csv"  
df = pd.read_csv(csv_path)

db_path = os.path.join("Datasets/spider/database", 'concert_singer', 'concert_singer' + ".sqlite")

df["SYM_DIFF_Without"] = df.apply(lambda row: execute_symmetric_difference(row["Gold"], row["Without Description"], db_path), axis=1)
df["SYM_DIFF_With"] = df.apply(lambda row: execute_symmetric_difference(row["Gold"], row["With Description"], db_path), axis=1)


updated_csv_path = "Description-based Analysis/AsymDIff/yicoder_desc_analysis_results.csv"
df.to_csv(updated_csv_path, index=False)

print(f"Updated CSV saved as {updated_csv_path}")
