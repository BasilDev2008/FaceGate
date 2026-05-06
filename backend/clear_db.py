import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "facegate.db")

print("Using database:", DB_PATH)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("DELETE FROM users;")
conn.commit()

cursor.execute("SELECT COUNT(*) FROM users;")
count = cursor.fetchone()[0]

conn.close()

print("Users left:", count)
print("Database cleared.")