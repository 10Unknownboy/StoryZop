import sqlite3
import json
import argparse
from pathlib import Path

def print_table(cursor, table_name):
    print(f"\n{'='*50}")
    print(f"TABLE: {table_name}")
    print(f"{'='*50}")
    
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    
    if not rows:
        print("(Empty)")
        return
        
    # Get column names
    col_names = [description[0] for description in cursor.description]
    
    for row in rows:
        print("-" * 30)
        for i, col in enumerate(col_names):
            val = row[i]
            # Try to format JSON if it looks like JSON
            if isinstance(val, str) and (val.startswith('{') or val.startswith('[')):
                try:
                    parsed = json.loads(val)
                    val = "\n" + json.dumps(parsed, indent=2)
                except json.JSONDecodeError:
                    pass
            print(f"{col}: {val}")

def main():
    parser = argparse.ArgumentParser(description="View StoryZop database contents")
    parser.add_argument("--db", default="data/storyzop.db", help="Path to database file")
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall() if r[0] != 'sqlite_sequence']
    
    print(f"Found {len(tables)} tables in {db_path.name}")
    for table in tables:
        print_table(cursor, table)
        
    conn.close()

if __name__ == "__main__":
    main()
