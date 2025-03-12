import sqlite3

def create_dummy_db():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    # cursor.execute("CREATE TABLE items (id INTEGER, fruit TEXT)")
    # # Insert sample data: note that 'apple' appears twice.
    # data = [
    #     (1, 'apple'),
    #     (2, 'banana'),
    #     (3, 'apple'),
    #     (4, 'cherry')
    # ]
    # cursor.execute("CREATE TABLE employee (id INTEGER, salary INTEGER)")
    # data = [
    #     (1, 100),
    #     (2, 1200),
    #     (3, 300),
    #     (4, 50)
    # ]
    cursor.execute("CREATE TABLE sales (product TEXT, qty INTEGER, price INTEGER)")
    data = [
        ("Laptop", 2, 1000),
        ("Mouse", 5, 50),
        ("Laptop", 2, 1000)
    ]
    # cursor.executemany("INSERT INTO items (id, fruit) VALUES (?, ?)", data)
    cursor.executemany("INSERT INTO sales (product, qty, price) VALUES (?, ?, ?)", data)
    # cursor.executemany("INSERT INTO employee (id, salary) VALUES (?, ?)", data)
    conn.commit()
    return conn

def symmetric_difference_query(conn, sql_two, sql_one):
    cursor = conn.cursor()
    sym_diff_sql = f"""
    {sql_one}
    UNION
    {sql_two}
    EXCEPT
    {sql_one}
    INTERSECT
    {sql_two};
    """
    sym_diff_sql = f"""
    {sql_two}
    EXCEPT
    {sql_one}
    UNION ALL
    {sql_one}
    EXCEPT
    {sql_two};
    """
    # sym_diff_sql = f"""
    # SELECT * FROM ({sql_one})
    # EXCEPT
    # SELECT * FROM ({sql_two})
    # UNION ALL
    # SELECT * FROM ({sql_two})
    # EXCEPT
    # SELECT * FROM ({sql_one})
    # """
        
    # sym_diff_sql = f"""
    # WITH pred AS (
    # SELECT fruit, COUNT(*) AS cnt
    # FROM Items
    # WHERE id <> 3
    # GROUP BY fruit
    # ),
    # gold AS (
    #     SELECT fruit, COUNT(*) AS cnt
    #     FROM Items
    #     GROUP BY fruit
    # )
    # SELECT * FROM pred
    # EXCEPT
    # SELECT * FROM gold
    # UNION
    # SELECT * FROM gold
    # EXCEPT
    # SELECT * FROM pred;
    # """
    
    
    print("Symmetric difference query:")
    print(sym_diff_sql)
    cursor.execute(sym_diff_sql)
    res = cursor.fetchall()
    return res

def main():
    conn = create_dummy_db()

    # sql_one = "SELECT fruit FROM Items"
    # sql_two = "SELECT fruit FROM Items WHERE id <> 3"
    
    sql_one = "SELECT DISTINCT product FROM sales"
    sql_two = "SELECT product FROM sales"
    
    # sql_one = "SELECT * FROM employee"
    # sql_two = "SELECT * FROM employee ORDER BY salary"
    
    # sql_one = "SELECT fruit, COUNT(*) AS cnt FROM Items WHERE id <> 3 GROUP BY fruit"
    # sql_two = "SELECT fruit, COUNT(*) AS cnt FROM Items GROUP BY fruit"
    
    cursor = conn.cursor()
    cursor.execute(sql_one)
    result_one = cursor.fetchall()
    print("Query One Result:")
    print(result_one) 
    
    cursor.execute(sql_two)
    result_two = cursor.fetchall()
    print("Query Two Result:")
    print(result_two)  
    
    # having the same cardinality is not enough to say both are same!
    # print(Counter(result_one) == Counter(result_two))
    
    print("Query One Result as SETS:")
    print(set(result_one)) 
    print("Query Two Result as SETS:")
    print(set(result_two))  
    if set(result_one) == set(result_two): 
        print('Set difference is equal!')
    else: 
        print('Set difference is not equal!')
        
    # print('Inbuilt sym diff:')
    # print(set(result_one).symmetric_difference(set(result_two)))
    
    sym_diff = symmetric_difference_query(conn, sql_two, sql_one)
    print("Symmetric difference result:")
    print(sym_diff) 
    
    if sym_diff:
        print("Queries are NOT equivalent (symmetric difference found)")
    else:
        print("Queries are equivalent")
    
    conn.close()

if __name__ == "__main__":
    main()
