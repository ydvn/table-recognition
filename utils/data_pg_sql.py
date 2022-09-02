import json
import sqlite3
import traceback

json_content = []

################## Read Data from JSON #############

with open("PubTabNet/pubtabnet/PubTabNet_2.0.0.jsonl", "r") as f:
    f_lines = f.readlines()
    for line in f_lines:
        json_content.append(json.loads(line))

################## Create Table #############

conn = sqlite3.connect("label_database.db")
c = conn.cursor()

c.execute(
    """
          CREATE TABLE IF NOT EXISTS labels 
          (
            [uuid] TEXT PRIMARY KEY,
            [filename] TEXT,
            [split] TEXT,
            [imgid] INTEGER,
            [html] TEXT
            )
          """
)

conn.commit()
conn.close()

################## Write Data to PgSQL Table #############

print(len(json_content))
total = len(json_content)
batch = 100
for i in range(0, len(json_content), batch):
    # if i>0:
    #     break
    print(f"{i+1}:{i+100}/{total}:")
    data = json_content[i : i + 100]
    values = []
    for j, data_point in enumerate(data):

        file_name = data_point.get("filename")
        split = data_point.get("split")
        imgid = data_point.get("imgid")
        uuid = f"{split}_{imgid}"
        html_str = json.dumps(data_point.get("html"))
        # html_str="Null"

        values.append((uuid, file_name, split, imgid, html_str))

    conn = sqlite3.connect("label_database.db")
    try:
        insert_query = """INSERT INTO labels 
        (uuid, filename, split, imgid, html)
        VALUES (?, ?, ?, ?, ?);"""
        c = conn.cursor()

        c.executemany(insert_query, values)
        print("Total", c.rowcount, "Records inserted successfully into labels table")
        conn.commit()
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        conn.close()
