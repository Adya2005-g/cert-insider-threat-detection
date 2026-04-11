import sqlite3
import os

# Check both potential paths
paths = ['instance/insider_threat.db', 'instance/users.db']

for p in paths:
    if os.path.exists(p):
        print(f"Checking {p}...")
        conn = sqlite3.connect(p)
        cursor = conn.cursor()
        
        # Check if table logs exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='logs'")
        if cursor.fetchone():
            try:
                cursor.execute("ALTER TABLE logs ADD COLUMN night_login_count FLOAT DEFAULT 0.0")
                cursor.execute("ALTER TABLE logs ADD COLUMN email_activity_count FLOAT DEFAULT 0.0")
                cursor.execute("ALTER TABLE logs ADD COLUMN usb_usage_count FLOAT DEFAULT 0.0")
                conn.commit()
                print(f"Successfully updated {p}")
            except sqlite3.OperationalError as e:
                print(f"Notice: {e}")
        conn.close()
