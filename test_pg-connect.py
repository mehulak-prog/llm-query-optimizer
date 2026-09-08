import psycopg2
conn = psycopg2.connect(host="localhost", port=5432,
                        dbname="sandbox", user="postgres", password="postgres", )
cur = conn.cursor()
cur.execute("SELECT version();")
print(cur.fetchone())
conn.close()
print("Connected successfully.")
