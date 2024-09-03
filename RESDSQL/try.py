import sqlite3

conn = sqlite3.connect('../Datasets/bird/databases/toxicology/toxicology.sqlite')
cursor = conn.cursor()
cursor.execute("select atom.element from atom where atom.atom_id = 'TRA0000_1'")

print('sairam')
rows = cursor.fetchall()
print(rows)
for row in rows:
    print(row)

conn.commit()
conn.close()
