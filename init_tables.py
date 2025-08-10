#!/usr/bin/env python3
"""
Initialize database tables
"""

import sqlite3

DB_PATH = './papers_database.db'

def init_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if email_queue table exists
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='email_queue'
    """)
    
    if cursor.fetchone() is None:
        print("Creating email_queue table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_queue (
                queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id TEXT NOT NULL,
                topic_name TEXT NOT NULL,
                recipient_email TEXT NOT NULL,
                paper_ids TEXT NOT NULL,
                html_body TEXT NOT NULL,
                text_body TEXT NOT NULL,
                subject TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                attempts INTEGER DEFAULT 0,
                last_error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                sent_at TEXT,
                postmark_message_id TEXT
            )
        ''')
        conn.commit()
        print("email_queue table created successfully!")
    else:
        print("email_queue table already exists")
    
    # Also ensure topic_id column exists in papers table
    cursor.execute("PRAGMA table_info(papers)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'topic_id' not in columns:
        print("Adding topic_id column to papers table...")
        cursor.execute("ALTER TABLE papers ADD COLUMN topic_id TEXT")
        conn.commit()
        print("topic_id column added successfully!")
    
    conn.close()
    print("\nDatabase initialization complete!")

if __name__ == "__main__":
    init_tables()