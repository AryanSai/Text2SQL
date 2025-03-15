import sqlite3, duckdb

def create_dummy_db():
    conn = duckdb.connect(database=':memory:')
    cursor = conn.cursor()

    # Create table and insert sample data
    cursor.execute("""
        CREATE TABLE sales (
            product TEXT,
            qty INTEGER,
            price INTEGER
        )
    """)

    data_sales = [
        ("Laptop", 2, 1000),
        ("Mouse", 5, 50),
        ("Laptop", 2, 1000)
    ]

    cursor.executemany(
        "INSERT INTO sales (product, qty, price) VALUES (?, ?, ?)", data_sales
    )

    conn.commit()
    return conn

def symmetric_difference_query(conn, sql_two, sql_one):
    cursor = conn.cursor()
    sym_diff_sql_1 = f"""
    {sql_two}
    EXCEPT ALL
    {sql_one};
    """

    print("\n[1] sql_two EXCEPT ALL sql_one:")
    print(sym_diff_sql_1)
    cursor.execute(sym_diff_sql_1)
    res1 = cursor.fetchall()
    print(res1)

    sym_diff_sql_2 = f"""
    {sql_one}
    EXCEPT ALL
    {sql_two};
    """

    print("\n[2] sql_one EXCEPT ALL sql_two:")
    print(sym_diff_sql_2)
    cursor.execute(sym_diff_sql_2)
    res2 = cursor.fetchall()
    print(res2)
    
    sym_diff_sql = f"""
    ({sql_two}
    EXCEPT ALL
    {sql_one})
    UNION ALL
    ({sql_one}
    EXCEPT ALL
    {sql_two});
    """

    print("\n[3] Symmetric difference query (UNION ALL of both EXCEPT ALLs):")
    print(sym_diff_sql)
    cursor.execute(sym_diff_sql)
    res = cursor.fetchall()

    return res

def main():
    conn = create_dummy_db()

    sql_one = "SELECT DISTINCT product FROM sales"
    sql_two = "SELECT product FROM sales"

    # sql_one = "SELECT * FROM sales"
    # sql_two = "SELECT * FROM sales ORDER BY qty"
    # sql_two = "(SELECT * FROM sales ORDER BY qty LIMIT ALL)"

    cursor = conn.cursor()

    cursor.execute(sql_one)
    result_one = cursor.fetchall()
    print("\nQuery One Result:")
    print(result_one)

    cursor.execute(sql_two)
    result_two = cursor.fetchall()
    print("\nQuery Two Result:")
    print(result_two)

    if result_one == result_two:
        print('\n✅ Normal difference is equal!')
    else:
        print('\n❌ Normal difference is not equal!')
        
    print("\nQuery One Result as SETS:")
    print(set(result_one))
    print("Query Two Result as SETS:")
    print(set(result_two))

    if set(result_one) == set(result_two):
        print('\n✅ Set difference is equal! (but this ignores duplicates!)')
    else:
        print('\n❌ Set difference is not equal!')

    sym_diff = symmetric_difference_query(conn, sql_two, sql_one)
    print("\nSymmetric difference result:")
    print(sym_diff)

    if sym_diff:
        print("\n❌ Queries are NOT equivalent (symmetric difference found)")
    else:
        print("\n✅ Queries are equivalent (no symmetric difference found)")

    conn.close()

if __name__ == "__main__":
    main()
