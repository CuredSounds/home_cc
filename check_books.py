import sqlite3

with sqlite3.connect("books.db") as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM books;")
    total = cursor.fetchone()[0]
    print(f"Total stored titles: {total}")

    cursor.execute(
        "SELECT clean_title, raw_title FROM books ORDER BY id LIMIT 5;"
    )
    for clean, raw in cursor.fetchall():
        print(f"- {clean} (from: {raw})")