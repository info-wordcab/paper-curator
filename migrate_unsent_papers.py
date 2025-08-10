#!/usr/bin/env python3
"""
One-time script to migrate unsent papers to the email queue
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

# Database path
DB_PATH = './papers_database.db'

def get_unsent_papers_by_topic(conn):
    """Get all unsent papers grouped by topic"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 
            topic_id,
            COUNT(*) as count
        FROM papers 
        WHERE email_sent_date IS NULL 
            AND original_summary IS NOT NULL
            AND topic_id IS NOT NULL
        GROUP BY topic_id
    ''')
    
    results = cursor.fetchall()
    print(f"\nFound unsent papers for {len(results)} topics:")
    for topic_id, count in results:
        print(f"  - {topic_id}: {count} papers")
    
    return results

def get_papers_for_topic(conn, topic_id, min_score=7.0):
    """Get all unsent papers for a specific topic"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM papers 
        WHERE email_sent_date IS NULL 
            AND original_summary IS NOT NULL
            AND relevance_score >= ?
            AND topic_id = ?
        ORDER BY relevance_score DESC
    ''', (min_score, topic_id))
    
    columns = [description[0] for description in cursor.description]
    papers = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return papers

def format_email_html(papers, topic_name):
    """Format papers as HTML (simplified version)"""
    html = f"""<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
        .paper {{ margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .paper-number {{ font-weight: bold; color: #333; font-size: 18px; }}
        .paper-title {{ color: #0066cc; font-size: 16px; margin: 10px 0; }}
        .paper-summary {{ margin: 10px 0; }}
        .paper-links {{ margin-top: 10px; font-size: 14px; }}
        .paper-links a {{ margin-right: 15px; }}
    </style>
</head>
<body>
    <h2>{topic_name} - ArXiv Research Digest (Recovery)</h2>
    <h3>{datetime.now().strftime('%B %d, %Y')}</h3>
    <p><em>This email contains previously processed papers that failed to send.</em></p>
"""
    
    for i, paper in enumerate(papers, 1):
        github_urls = json.loads(paper.get('github_urls', '[]'))
        github_html = ""
        if github_urls:
            github_html = " | ".join([f'<a href="{url}">GitHub</a>' for url in github_urls[:2]])
        
        html += f"""
        <div class="paper">
            <div class="paper-number">#{i}</div>
            <div class="paper-title">{paper['title']}</div>
            <div class="paper-summary">{paper.get('original_summary', paper['abstract'][:300] + '...')}</div>
            <div class="paper-links">
                <a href="{paper['pdf_url']}">PDF</a> |
                <a href="{paper['arxiv_id']}">arXiv</a>
                {' | ' + github_html if github_html else ''}
            </div>
        </div>
        """
    
    html += "</body></html>"
    return html

def format_email_text(papers, topic_name):
    """Format papers as plain text"""
    text = f"{topic_name} - ArXiv Research Digest (Recovery)\n"
    text += f"{datetime.now().strftime('%B %d, %Y')}\n"
    text += "=" * 60 + "\n\n"
    text += "This email contains previously processed papers that failed to send.\n\n"
    
    for i, paper in enumerate(papers, 1):
        text += f"#{i}. {paper['title']}\n"
        text += f"{paper.get('original_summary', paper['abstract'][:300] + '...')}\n"
        text += f"PDF: {paper['pdf_url']}\n"
        
        github_urls = json.loads(paper.get('github_urls', '[]'))
        if github_urls:
            text += f"Code: {github_urls[0]}\n"
        
        text += "\n" + "-" * 40 + "\n\n"
    
    return text

def queue_emails_for_topic(conn, topic_id, topic_config, papers):
    """Queue email for a topic's unsent papers"""
    cursor = conn.cursor()
    
    paper_ids = [p['paper_id'] for p in papers]
    subject = f"[ArXiv Digest] {topic_config['name']} - {datetime.now().strftime('%Y-%m-%d')} (Recovery)"
    html_body = format_email_html(papers, topic_config['name'])
    text_body = format_email_text(papers, topic_config['name'])
    
    cursor.execute('''
        INSERT INTO email_queue (
            topic_id, topic_name, recipient_email, paper_ids,
            subject, html_body, text_body
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        topic_id, 
        topic_config['name'], 
        topic_config['email'],
        json.dumps(paper_ids),
        subject, 
        html_body, 
        text_body
    ))
    
    queue_id = cursor.lastrowid
    conn.commit()
    
    print(f"  Created queue entry {queue_id} for {len(papers)} papers")
    return queue_id

def main():
    """Main migration function"""
    # Topic configurations - update these based on your topics.yaml
    TOPIC_CONFIGS = {
        'speech_processing': {
            'name': 'Speech Processing Research',
            'email': 'aleks@wordcab.com',
            'min_score': 7
        },
        'neural_speech_decoding': {
            'name': 'Neural Speech Decoding',
            'email': 'neurotech@example.com',
            'min_score': 6
        }
    }
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get unsent papers by topic
    topic_counts = get_unsent_papers_by_topic(conn)
    
    if not topic_counts:
        print("\nNo unsent papers found!")
        return
    
    # Process each topic
    print("\nMigrating papers to email queue...")
    for topic_id, _ in topic_counts:
        if topic_id not in TOPIC_CONFIGS:
            print(f"\nWARNING: No config found for topic '{topic_id}' - skipping")
            continue
        
        config = TOPIC_CONFIGS[topic_id]
        print(f"\nProcessing {topic_id}:")
        
        # Get papers for this topic
        papers = get_papers_for_topic(conn, topic_id, config['min_score'])
        
        if papers:
            # Split into batches if needed (e.g., 20 papers per email)
            batch_size = 20
            for i in range(0, len(papers), batch_size):
                batch = papers[i:i+batch_size]
                queue_emails_for_topic(conn, topic_id, config, batch)
    
    # Show final status
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM email_queue WHERE status = 'pending'")
    pending_count = cursor.fetchone()[0]
    
    print(f"\nMigration complete! {pending_count} emails queued.")
    print("\nTo send these emails, run:")
    print("  python arxiv_curator.py --retry-emails")
    
    conn.close()

if __name__ == "__main__":
    main()