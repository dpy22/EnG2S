import sqlite3

# 查询目标数据库文件
db_path = r"D:\\能源系统\\EnG2S\\reports\\xgboost_run_20251220_104302\\optimization_history.db"

# 连接数据库
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查看数据库中所有表
print("数据库中所有表：")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
for t in tables:
    print(t[0])

print("\n每个表的前5条记录及结构：")
for t in tables:
    print(f"\n表名: {t[0]}")
    # 查询表结构
    cursor.execute(f"PRAGMA table_info('{t[0]}');")
    cols = cursor.fetchall()
    print("字段:")
    for col in cols:
        print(f"  {col[1]} ({col[2]})")
    # 查询前5条数据
    cursor.execute(f"SELECT * FROM '{t[0]}' LIMIT 5;")
    rows = cursor.fetchall()
    print("前5条记录:")
    for row in rows:
        print(row)

conn.close()