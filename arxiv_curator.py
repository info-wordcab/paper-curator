#!/usr/bin/env python3
"""
ArXiv Paper Curator
Automated system for collecting, filtering, summarizing, and distributing
relevant research papers from arXiv with feedback learning.
"""

import os
import sys
import json
import sqlite3
import logging
import argparse
import re
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from logging.handlers import RotatingFileHandler
import base64

# Third-party imports
import arxiv
import ezgmail
from anthropic import Anthropic
from postmarker.core import PostmarkClient
from dotenv import load_dotenv
import PyPDF2
import requests
import urllib.parse
import yaml

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration management class"""
    
    # API Keys
    CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
    POSTMARK_SERVER_TOKEN = os.getenv('POSTMARK_SERVER_TOKEN')
    POSTMARK_FROM_EMAIL = os.getenv('POSTMARK_FROM_EMAIL')
    DEFAULT_RECIPIENT_EMAIL = os.getenv('DEFAULT_RECIPIENT_EMAIL')
    
    # Database
    DB_PATH = os.getenv('DB_PATH', './papers_database.db')
    DB_BACKUP_PATH = os.getenv('DB_BACKUP_PATH', './backups/')
    
    # Topics Configuration
    TOPICS_FILE = os.getenv('TOPICS_FILE', './topics.yaml')
    TOPICS = {}
    TOPIC_DEFAULTS = {}
    
    @classmethod
    def load_topics(cls):
        """Load topics configuration from YAML file"""
        try:
            with open(cls.TOPICS_FILE, 'r') as f:
                config = yaml.safe_load(f)
                cls.TOPICS = config.get('topics', {})
                cls.TOPIC_DEFAULTS = config.get('defaults', {
                    'max_results': 30,
                    'days_back': 7,
                    'min_relevance_score': 7,
                    'schedule_days': ['Mon', 'Wed', 'Fri']
                })
                logger.info(f"Loaded {len(cls.TOPICS)} topics from {cls.TOPICS_FILE}")
        except Exception as e:
            logger.error(f"Error loading topics configuration: {e}")
            cls.TOPICS = {}
            cls.TOPIC_DEFAULTS = {}
    
    # Claude
    CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'claude-opus-4-20250514')
    CLAUDE_MAX_TOKENS = int(os.getenv('CLAUDE_MAX_TOKENS', '4000'))
    CLAUDE_TEMPERATURE = float(os.getenv('CLAUDE_TEMPERATURE', '0.3'))
    
    # Email
    EMAIL_SUBJECT_PREFIX = os.getenv('EMAIL_SUBJECT_PREFIX', '[ArXiv Digest]')
    MAX_PAPERS_PER_EMAIL = int(os.getenv('MAX_PAPERS_PER_EMAIL', '20'))
    MIN_PAPERS_TO_SEND = int(os.getenv('MIN_PAPERS_TO_SEND', '3'))
    REPLY_CHECK_DELAY_HOURS = int(os.getenv('REPLY_CHECK_DELAY_HOURS', '24'))
    
    # Paths
    PDF_STORAGE_PATH = os.getenv('PDF_STORAGE_PATH', './pdfs/')
    TEMP_PATH = os.getenv('TEMP_PATH', './temp/')
    LOG_FILE = os.getenv('LOG_FILE', './logs/arxiv_curator.log')
    
    # Features
    ENABLE_CODE_DETECTION = os.getenv('ENABLE_CODE_DETECTION', 'true').lower() == 'true'
    ENABLE_PDF_DOWNLOAD = os.getenv('ENABLE_PDF_DOWNLOAD', 'true').lower() == 'true'
    ENABLE_REPLY_PROCESSING = os.getenv('ENABLE_REPLY_PROCESSING', 'true').lower() == 'true'
    DRY_RUN = os.getenv('DRY_RUN', 'false').lower() == 'true'
    
    # Performance
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))
    REQUEST_DELAY_SECONDS = float(os.getenv('REQUEST_DELAY_SECONDS', '3.0'))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging with rotation"""
    log_dir = Path(Config.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler = RotatingFileHandler(
        Config.LOG_FILE,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, os.getenv('LOG_LEVEL', 'INFO')))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# ============================================================================
# DATABASE MANAGEMENT
# ============================================================================

class Database:
    """Database management class"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Papers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                arxiv_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                authors TEXT,
                published_date TEXT,
                categories TEXT,
                abstract TEXT,
                pdf_url TEXT,
                has_code INTEGER DEFAULT 0,
                github_urls TEXT,
                original_summary TEXT,
                edited_summary TEXT,
                relevance_score REAL,
                selected INTEGER DEFAULT 0,
                email_sent_date TEXT,
                topic_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Summary edits table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS summary_edits (
                edit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT,
                before_text TEXT,
                after_text TEXT,
                edit_date TEXT DEFAULT CURRENT_TIMESTAMP,
                used_in_prompts INTEGER DEFAULT 0,
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
            )
        ''')
        
        # Email logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_logs (
                email_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sent_date TEXT DEFAULT CURRENT_TIMESTAMP,
                paper_ids TEXT,
                postmark_message_id TEXT,
                reply_received INTEGER DEFAULT 0,
                reply_processed INTEGER DEFAULT 0,
                reply_content TEXT
            )
        ''')
        
        # Email queue table for retry capability
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
        conn.close()
        logger.info("Database initialized successfully")
    
    def paper_exists(self, arxiv_id: str) -> bool:
        """Check if paper already exists in database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM papers WHERE arxiv_id = ?', (arxiv_id,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    
    def insert_paper(self, paper_data: Dict[str, Any]):
        """Insert new paper into database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO papers (
                paper_id, arxiv_id, title, authors, published_date,
                categories, abstract, pdf_url, topic_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            paper_data['paper_id'],
            paper_data['arxiv_id'],
            paper_data['title'],
            paper_data['authors'],
            paper_data['published_date'],
            paper_data['categories'],
            paper_data['abstract'],
            paper_data['pdf_url'],
            paper_data.get('topic_id'),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_unsent_papers(self, topic_id: str = None, limit: int = 20, min_score: float = 7.0) -> List[Dict[str, Any]]:
        """Get papers that haven't been sent yet, optionally filtered by topic"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if topic_id:
            cursor.execute('''
                SELECT * FROM papers 
                WHERE email_sent_date IS NULL 
                AND original_summary IS NOT NULL
                AND relevance_score >= ?
                AND topic_id = ?
                ORDER BY relevance_score DESC
                LIMIT ?
            ''', (min_score, topic_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM papers 
                WHERE email_sent_date IS NULL 
                AND original_summary IS NOT NULL
                AND relevance_score >= ?
                ORDER BY relevance_score DESC
                LIMIT ?
            ''', (min_score, limit))
        
        columns = [description[0] for description in cursor.description]
        papers = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        
        return papers
    
    def get_edit_examples(self, limit: int = 10) -> List[Tuple[str, str]]:
        """Get recent edit examples for prompt enhancement"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT before_text, after_text 
            FROM summary_edits 
            WHERE used_in_prompts = 1
            ORDER BY edit_date DESC
            LIMIT ?
        ''', (limit,))
        
        examples = cursor.fetchall()
        conn.close()
        
        return examples
    
    def mark_papers_sent(self, paper_ids: List[str], message_id: str):
        """Mark papers as sent and log email"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Update papers
        cursor.executemany(
            'UPDATE papers SET email_sent_date = ? WHERE paper_id = ?',
            [(datetime.now().isoformat(), pid) for pid in paper_ids]
        )
        
        # Log email
        cursor.execute('''
            INSERT INTO email_logs (paper_ids, postmark_message_id)
            VALUES (?, ?)
        ''', (json.dumps(paper_ids), message_id))
        
        conn.commit()
        conn.close()
    
    def queue_email(self, topic_id: str, topic_name: str, recipient_email: str, 
                   paper_ids: List[str], subject: str, html_body: str, text_body: str) -> int:
        """Queue an email for sending/retry"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO email_queue (
                topic_id, topic_name, recipient_email, paper_ids,
                subject, html_body, text_body
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            topic_id, topic_name, recipient_email, json.dumps(paper_ids),
            subject, html_body, text_body
        ))
        
        queue_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return queue_id
    
    def get_pending_emails(self, max_attempts: int = 3) -> List[Dict[str, Any]]:
        """Get pending emails from queue"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM email_queue 
            WHERE status = 'pending' 
            AND attempts < ?
            ORDER BY created_at ASC
        ''', (max_attempts,))
        
        columns = [description[0] for description in cursor.description]
        emails = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        
        return emails
    
    def update_email_queue(self, queue_id: int, status: str, message_id: str = None, error: str = None):
        """Update email queue status"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if status == 'sent':
            cursor.execute('''
                UPDATE email_queue 
                SET status = ?, postmark_message_id = ?, sent_at = ?
                WHERE queue_id = ?
            ''', (status, message_id, datetime.now().isoformat(), queue_id))
        else:
            cursor.execute('''
                UPDATE email_queue 
                SET status = ?, last_error = ?, attempts = attempts + 1
                WHERE queue_id = ?
            ''', (status, error, queue_id))
        
        conn.commit()
        conn.close()

# ============================================================================
# ARXIV INTERFACE
# ============================================================================

class ArxivManager:
    """Manage arXiv API interactions"""
    
    def __init__(self, db: Database):
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=Config.REQUEST_DELAY_SECONDS,
            num_retries=Config.MAX_RETRIES
        )
        self.db = db
    
    def search_papers(self, search_term: str, days_back: int = 7, max_results: int = 30) -> List[arxiv.Result]:
        """Search for recent papers matching term"""
        try:
            # Build date-restricted query
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
            query = f'{search_term} AND submittedDate:[{start_date} TO *]'
            
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            results = list(self.client.results(search))
            logger.info(f"Found {len(results)} papers for term: {search_term}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []

    def search_papers_via_advanced_url(self, advanced_url: str, days_back: int = 7, max_results: int = 30) -> List[arxiv.Result]:
        """Search papers by parsing an arXiv Advanced Search URL and translating to API query.

        Only the date range is made dynamic (from now - days_back to now). Page size is ignored.
        """
        try:
            # Allow users to prefix the URL with '@' (for convenience in tooling)
            advanced_url = (advanced_url or '').lstrip('@').strip()
            parsed = urllib.parse.urlparse(advanced_url)
            params = urllib.parse.parse_qs(parsed.query)

            # Build boolean query from terms
            # Expected params: terms-<n>-operator, terms-<n>-term, terms-<n>-field
            # Operators are relative to previous term.
            term_indices = []
            for key in params.keys():
                m = re.match(r"terms-(\d+)-term", key)
                if m:
                    term_indices.append(int(m.group(1)))
            term_indices = sorted(set(term_indices))

            def map_field(field_value: str) -> str:
                field_value = (field_value or '').lower()
                if field_value == 'title':
                    return 'ti'
                if field_value == 'abstract':
                    return 'abs'
                if field_value == 'author' or field_value == 'authors':
                    return 'au'
                if field_value in ('comments', 'journal reference', 'acm classification', 'msc classification', 'report number', 'arxiv identifier', 'doi', 'all fields'):
                    return 'all'
                return 'all'

            clauses: List[str] = []
            for idx in term_indices:
                term_list = params.get(f'terms-{idx}-term', [''])
                if not term_list or not term_list[0].strip():
                    continue
                term_value = term_list[0].strip()
                field_value = (params.get(f'terms-{idx}-field', ['all'])[0]).strip()
                field_tag = map_field(field_value)
                # Quote term; urllib.parse already decoded '+' to spaces
                # Escape internal quotes
                safe_term = term_value.replace('"', '\\"')
                clause = f'{field_tag}:"{safe_term}"' if field_tag != 'all' else f'"{safe_term}"'

                operator = (params.get(f'terms-{idx}-operator', ['AND'])[0]).upper()
                if not clauses:
                    clauses.append(f'({clause})')
                else:
                    if operator not in ('AND', 'OR', 'ANDNOT', 'NOT'):
                        operator = 'AND'
                    if operator == 'NOT':
                        operator = 'ANDNOT'
                    clauses.append(f'{operator} ({clause})')

            base_query = ' '.join(clauses).strip()
            if not base_query:
                logger.warning('Advanced URL produced empty query; falling back to empty results.')
                return []

            # Optional category scoping: if computer_science=y, we could add category filters.
            # We keep it simple and rely on terms. If needed, extend with explicit cs.* categories.

            # Dynamic date window
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
            end_date = datetime.now().strftime('%Y%m%d')
            date_filter = f'submittedDate:[{start_date} TO {end_date}]'

            final_query = f'({base_query}) AND {date_filter}'

            search = arxiv.Search(
                query=final_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            results = list(self.client.results(search))
            logger.info(f"Found {len(results)} papers via Advanced URL")
            return results
        except Exception as e:
            logger.error(f"Error parsing Advanced URL or searching arXiv: {e}")
            return []
    
    def collect_papers_for_topic(self, topic_id: str, topic_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect new papers for a specific topic"""
        all_papers = []
        
        # Get topic-specific settings with fallbacks to defaults
        max_results = topic_config.get('max_results', Config.TOPIC_DEFAULTS['max_results'])
        days_back = topic_config.get('days_back', Config.TOPIC_DEFAULTS['days_back'])
        
        searches = topic_config.get('searches', [])
        
        for search in searches:
            try:
                if 'url' in search:
                    # Advanced search URL
                    logger.info(f"Topic {topic_id}: Using advanced search URL")
                    results = self.search_papers_via_advanced_url(search['url'], days_back, max_results)
                elif 'query' in search:
                    # Simple keyword search
                    logger.info(f"Topic {topic_id}: Searching for '{search['query']}'")
                    results = self.search_papers(search['query'], days_back, max_results)
                else:
                    continue
                
                for paper in results:
                    if self.db.paper_exists(paper.entry_id):
                        continue
                    
                    paper_data = {
                        'paper_id': hashlib.md5(paper.entry_id.encode()).hexdigest(),
                        'arxiv_id': paper.entry_id,
                        'title': paper.title,
                        'authors': ', '.join([author.name for author in paper.authors]),
                        'published_date': paper.published.isoformat(),
                        'categories': ', '.join(paper.categories),
                        'abstract': paper.summary,
                        'pdf_url': paper.pdf_url,
                        'topic_id': topic_id
                    }
                    self.db.insert_paper(paper_data)
                    all_papers.append(paper_data)
                
                time.sleep(Config.REQUEST_DELAY_SECONDS)
                
            except Exception as e:
                logger.error(f"Error processing search for topic {topic_id}: {e}")
        
        logger.info(f"Collected {len(all_papers)} new papers for topic {topic_id}")
        return all_papers

# ============================================================================
# CLAUDE INTERFACE
# ============================================================================

class ClaudeManager:
    """Manage Claude API interactions"""
    
    def __init__(self):
        self.client = Anthropic(api_key=Config.CLAUDE_API_KEY)
    
    def filter_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter papers using Claude for relevance"""
        filtered = []
        
        # Load filter prompt
        filter_prompt = self._load_prompt('filter')
        
        for batch in self._batch_papers(papers, Config.BATCH_SIZE):
            try:
                # Prepare batch for Claude
                batch_text = self._format_papers_for_filtering(batch)
                
                response = self.client.messages.create(
                    model=Config.CLAUDE_MODEL,
                    max_tokens=Config.CLAUDE_MAX_TOKENS,
                    temperature=Config.CLAUDE_TEMPERATURE,
                    messages=[
                        {
                            "role": "user",
                            "content": f"{filter_prompt}\n\nPapers to evaluate:\n{batch_text}"
                        }
                    ]
                )
                
                # Parse response and update papers
                scores = self._parse_filter_response(response.content[0].text)
                for paper, score in zip(batch, scores):
                    paper['relevance_score'] = score
                    if score >= float(os.getenv('DEFAULT_MIN_RELEVANCE_SCORE', '7')):
                        filtered.append(paper)
                        self._update_paper_score(paper['paper_id'], score)
                
            except Exception as e:
                logger.error(f"Error filtering papers: {e}")
            
            time.sleep(Config.REQUEST_DELAY_SECONDS)
        
        logger.info(f"Filtered to {len(filtered)} relevant papers")
        return filtered
    
    def generate_summary(self, paper: Dict[str, Any], edit_examples: List[Tuple[str, str]]) -> str:
        """Generate enhanced summary for a paper"""
        try:
            # Load base prompt
            summary_prompt = self._load_prompt('summary')
            
            # Add edit examples if available
            if edit_examples:
                examples_text = self._format_edit_examples(edit_examples)
                summary_prompt += f"\n\nPrevious feedback examples:\n{examples_text}"
            
            # Download and extract paper content if enabled
            full_text = paper['abstract']
            if Config.ENABLE_PDF_DOWNLOAD:
                pdf_text = self._extract_pdf_text(paper['pdf_url'])
                if pdf_text:
                    full_text = pdf_text[:10000]  # Limit to prevent token overflow
            
            response = self.client.messages.create(
                model=Config.CLAUDE_MODEL,
                max_tokens=500,
                temperature=Config.CLAUDE_TEMPERATURE,
                messages=[
                    {
                        "role": "user",
                        "content": f"{summary_prompt}\n\nPaper Title: {paper['title']}\n\nContent:\n{full_text}"
                    }
                ]
            )
            
            summary = response.content[0].text.strip()
            
            # Update database
            self._update_paper_summary(paper['paper_id'], summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return paper['abstract'][:300] + "..."
    
    def _load_prompt(self, prompt_type: str) -> str:
        """Load prompt from file or use default"""
        prompt_path = os.getenv(f'CLAUDE_{prompt_type.upper()}_PROMPT_PATH')
        
        if prompt_path and os.path.exists(prompt_path):
            with open(prompt_path, 'r') as f:
                return f.read()
        
        # Default prompts
        if prompt_type == 'filter':
            return """Evaluate the following research papers for relevance to cutting-edge AI/ML research.
            Score each paper from 1-10 based on:
            - Novelty of approach
            - Practical applications
            - Quality of methodology
            - Availability of code/implementation
            
            Return scores in format: Paper 1: [score], Paper 2: [score], etc."""
        
        elif prompt_type == 'summary':
            return """Create a concise, informative summary (150-250 words) of this research paper.
            Focus on:
            - Key innovation or contribution
            - Methodology highlights
            - Main results and findings
            - Practical implications
            - Any limitations mentioned
            
            Write in an engaging, accessible style for technical readers."""
        
        return ""
    
    def _format_papers_for_filtering(self, papers: List[Dict[str, Any]]) -> str:
        """Format papers for batch filtering"""
        formatted = []
        for i, paper in enumerate(papers, 1):
            formatted.append(f"Paper {i}:\nTitle: {paper['title']}\nAbstract: {paper['abstract'][:500]}...")
        return "\n\n".join(formatted)
    
    def _parse_filter_response(self, response: str) -> List[float]:
        """Parse relevance scores from Claude response"""
        scores = []
        pattern = r'Paper \d+:\s*(\d+(?:\.\d+)?)'
        matches = re.findall(pattern, response)
        
        for match in matches:
            try:
                scores.append(float(match))
            except ValueError:
                scores.append(5.0)  # Default middle score
        
        return scores
    
    def _format_edit_examples(self, examples: List[Tuple[str, str]]) -> str:
        """Format edit examples for prompt"""
        formatted = []
        for before, after in examples[:5]:  # Limit to 5 examples
            formatted.append(f"BEFORE: {before[:200]}...\nAFTER: {after[:200]}...")
        return "\n\n".join(formatted)
    
    def _extract_pdf_text(self, pdf_url: str) -> Optional[str]:
        """Download and extract text from PDF"""
        try:
            # Download PDF
            response = requests.get(pdf_url, timeout=30)
            pdf_path = Path(Config.PDF_STORAGE_PATH) / f"{hashlib.md5(pdf_url.encode()).hexdigest()}.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            # Extract text
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages[:10]:  # First 10 pages
                    text += page.extract_text()
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return None
    
    def _update_paper_score(self, paper_id: str, score: float):
        """Update paper relevance score in database"""
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('UPDATE papers SET relevance_score = ? WHERE paper_id = ?', (score, paper_id))
        conn.commit()
        conn.close()
    
    def _update_paper_summary(self, paper_id: str, summary: str):
        """Update paper summary in database"""
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('UPDATE papers SET original_summary = ? WHERE paper_id = ?', (summary, paper_id))
        conn.commit()
        conn.close()
    
    def _batch_papers(self, papers: List[Dict[str, Any]], batch_size: int):
        """Yield batches of papers"""
        for i in range(0, len(papers), batch_size):
            yield papers[i:i + batch_size]

# ============================================================================
# CODE DETECTION
# ============================================================================

class CodeDetector:
    """Detect code availability in papers"""
    
    def __init__(self):
        self.github_pattern = re.compile(r'github\.com/[\w\-]+/[\w\-]+', re.IGNORECASE)
        self.code_keywords = [
            'github.com', 'code available', 'implementation',
            'repository', 'source code', 'open source',
            'reproducibility', 'gitlab.com', 'bitbucket.org'
        ]
    
    def detect_code_in_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect and extract code links from papers"""
        for paper in papers:
            try:
                has_code, github_urls = self._check_for_code(paper)
                paper['has_code'] = has_code
                paper['github_urls'] = github_urls
                
                # Update database
                self._update_paper_code_info(paper['paper_id'], has_code, github_urls)
                
            except Exception as e:
                logger.error(f"Error detecting code for paper {paper['paper_id']}: {e}")
        
        return papers
    
    def _check_for_code(self, paper: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if paper has associated code"""
        github_urls = []
        
        # Check abstract
        abstract_urls = self.github_pattern.findall(paper['abstract'])
        github_urls.extend([f"https://{url}" for url in abstract_urls])
        
        # Check for code keywords
        text_to_check = paper['abstract'].lower()
        has_code_keywords = any(keyword in text_to_check for keyword in self.code_keywords)
        
        # If PDF download is enabled, check PDF
        if Config.ENABLE_PDF_DOWNLOAD and paper.get('pdf_url'):
            pdf_urls = self._extract_urls_from_pdf(paper['pdf_url'])
            github_urls.extend(pdf_urls)
        
        # Validate GitHub URLs
        validated_urls = self._validate_github_urls(github_urls)
        
        has_code = len(validated_urls) > 0 or has_code_keywords
        
        return has_code, validated_urls
    
    def _extract_urls_from_pdf(self, pdf_url: str) -> List[str]:
        """Extract GitHub URLs from PDF"""
        try:
            response = requests.get(pdf_url, timeout=30)
            pdf_content = response.content.decode('utf-8', errors='ignore')
            
            urls = self.github_pattern.findall(pdf_content)
            return [f"https://{url}" for url in urls]
            
        except Exception as e:
            logger.error(f"Error extracting URLs from PDF: {e}")
            return []
    
    def _validate_github_urls(self, urls: List[str]) -> List[str]:
        """Validate that GitHub URLs exist"""
        validated = []
        for url in list(set(urls)):  # Remove duplicates
            try:
                response = requests.head(url, timeout=5, allow_redirects=True)
                if response.status_code < 400:
                    validated.append(url)
            except:
                pass
        
        return validated
    
    def _update_paper_code_info(self, paper_id: str, has_code: bool, github_urls: List[str]):
        """Update paper code information in database"""
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE papers SET has_code = ?, github_urls = ? WHERE paper_id = ?',
            (int(has_code), json.dumps(github_urls), paper_id)
        )
        conn.commit()
        conn.close()

# ============================================================================
# EMAIL MANAGEMENT
# ============================================================================

class EmailManager:
    """Manage email sending and reply processing"""
    
    def __init__(self, db: Database):
        self.postmark = PostmarkClient(server_token=Config.POSTMARK_SERVER_TOKEN)
        self.db = db
        
        # Initialize ezgmail if enabled
        if Config.ENABLE_REPLY_PROCESSING:
            try:
                ezgmail.init(tokenFile="./token.json", credentialsFile="./credentials.json")
                self.gmail_enabled = True
            except Exception as e:
                logger.error(f"Failed to initialize ezgmail: {e}")
                self.gmail_enabled = False
    
    def send_digest(self, papers: List[Dict[str, Any]], topic_id: str, topic_name: str, recipient_email: str) -> bool:
        """Send email digest of papers for a specific topic"""
        if not papers:
            logger.info(f"No papers to send for topic {topic_name}")
            return False
        
        # Topic-specific minimum papers check could be added here
        min_papers = Config.MIN_PAPERS_TO_SEND
        if len(papers) < min_papers:
            logger.info(f"Only {len(papers)} papers for {topic_name}, minimum is {min_papers}")
            return False
        
        # Format email
        subject = f"{Config.EMAIL_SUBJECT_PREFIX} {topic_name} - {datetime.now().strftime('%Y-%m-%d')}"
        html_body = self._format_email_html(papers, topic_name)
        text_body = self._format_email_text(papers, topic_name)
        paper_ids = [p['paper_id'] for p in papers]
        
        # Queue the email
        queue_id = self.db.queue_email(
            topic_id=topic_id,
            topic_name=topic_name,
            recipient_email=recipient_email,
            paper_ids=paper_ids,
            subject=subject,
            html_body=html_body,
            text_body=text_body
        )
        
        logger.info(f"Queued email for {topic_name} with {len(papers)} papers (queue_id: {queue_id})")
        
        # Try to send immediately
        return self._send_queued_email({
            'queue_id': queue_id,
            'recipient_email': recipient_email,
            'subject': subject,
            'html_body': html_body,
            'text_body': text_body,
            'paper_ids': json.dumps(paper_ids)
        })
    
    def _send_queued_email(self, email_data: Dict[str, Any]) -> bool:
        """Send a queued email"""
        try:
            if Config.DRY_RUN:
                logger.info(f"DRY RUN: Would send email to {email_data['recipient_email']}")
                self.db.update_email_queue(email_data['queue_id'], 'sent', 'dry-run-message-id')
                return True
            
            # Send email
            response = self.postmark.emails.send(
                From=Config.POSTMARK_FROM_EMAIL,
                To=email_data['recipient_email'],
                Subject=email_data['subject'],
                HtmlBody=email_data['html_body'],
                TextBody=email_data['text_body']
            )
            
            # Mark as sent
            paper_ids = json.loads(email_data['paper_ids'])
            self.db.mark_papers_sent(paper_ids, response['MessageID'])
            self.db.update_email_queue(email_data['queue_id'], 'sent', response['MessageID'])
            
            logger.info(f"Sent email to {email_data['recipient_email']}. Message ID: {response['MessageID']}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error sending email to {email_data['recipient_email']}: {error_msg}")
            self.db.update_email_queue(email_data['queue_id'], 'pending', error=error_msg)
            return False
    
    def send_pending_emails(self):
        """Process all pending emails in the queue"""
        pending_emails = self.db.get_pending_emails()
        
        if not pending_emails:
            logger.info("No pending emails in queue")
            return
        
        logger.info(f"Processing {len(pending_emails)} pending emails")
        
        for email in pending_emails:
            self._send_queued_email(email)
            time.sleep(1)  # Rate limiting
    
    def check_replies(self):
        """Check for and process email replies"""
        if not self.gmail_enabled:
            logger.warning("Gmail not enabled, skipping reply check")
            return
        
        try:
            # Search for unread replies
            threads = ezgmail.search(f'subject:"{Config.EMAIL_SUBJECT_PREFIX}" is:unread')
            
            for thread in threads:
                self._process_reply_thread(thread)
            
        except Exception as e:
            logger.error(f"Error checking replies: {e}")
    
    def _format_email_html(self, papers: List[Dict[str, Any]], topic_name: str) -> str:
        """Format papers as HTML email"""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .paper {{ margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .paper-number {{ font-weight: bold; color: #333; font-size: 18px; }}
                .paper-title {{ color: #0066cc; font-size: 16px; margin: 10px 0; }}
                .paper-summary {{ margin: 10px 0; }}
                .paper-links {{ margin-top: 10px; font-size: 14px; }}
                .paper-links a {{ margin-right: 15px; }}
                .instructions {{ background: #f0f0f0; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h2>{topic_name} - ArXiv Research Digest</h2>
            <h3>{datetime.now().strftime('%B %d, %Y')}</h3>
            
            <div class="instructions">
                <strong>To provide feedback:</strong> Reply with paper numbers (e.g., "1-3, 5, 7") to select papers,
                followed by any edits to specific summaries. Format edits as "Paper 3: [your edited summary]"
            </div>
        """
        
        for i, paper in enumerate(papers, 1):
            github_links = json.loads(paper.get('github_urls', '[]'))
            github_html = ""
            if github_links:
                github_html = " | ".join([f'<a href="{url}">GitHub</a>' for url in github_links[:2]])
            
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
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _format_email_text(self, papers: List[Dict[str, Any]], topic_name: str) -> str:
        """Format papers as plain text email"""
        text = f"{topic_name} - ArXiv Research Digest\n"
        text += f"{datetime.now().strftime('%B %d, %Y')}\n"
        text += "=" * 60 + "\n\n"
        text += "Reply with paper numbers to select, and any summary edits.\n\n"
        
        for i, paper in enumerate(papers, 1):
            text += f"#{i}. {paper['title']}\n"
            text += f"{paper.get('original_summary', paper['abstract'][:300] + '...')}\n"
            text += f"PDF: {paper['pdf_url']}\n"
            
            github_links = json.loads(paper.get('github_urls', '[]'))
            if github_links:
                text += f"Code: {github_links[0]}\n"
            
            text += "\n" + "-" * 40 + "\n\n"
        
        return text
    
    def _process_reply_thread(self, thread):
        """Process a reply email thread"""
        try:
            latest_message = thread.messages[-1]
            
            # Skip if from ourselves
            if Config.POSTMARK_FROM_EMAIL in latest_message.sender:
                return
            
            content = latest_message.body
            
            # Extract selections
            selections = self._parse_selections(content)
            
            # Extract edits
            edits = self._parse_edits(content)
            
            # Update database
            if selections:
                self._update_selected_papers(selections)
            
            if edits:
                self._save_edits(edits)
            
            # Mark as read
            thread.markAsRead()
            
            logger.info(f"Processed reply with {len(selections)} selections and {len(edits)} edits")
            
        except Exception as e:
            logger.error(f"Error processing reply: {e}")
    
    def _parse_selections(self, content: str) -> List[int]:
        """Parse paper selections from reply"""
        selections = []
        
        # Look for patterns like "1-3, 5, 7-9"
        pattern = r'(\d+)(?:-(\d+))?'
        matches = re.findall(pattern, content)
        
        for match in matches:
            if match[1]:  # Range
                selections.extend(range(int(match[0]), int(match[1]) + 1))
            else:  # Single number
                selections.append(int(match[0]))
        
        return list(set(selections))
    
    def _parse_edits(self, content: str) -> Dict[int, str]:
        """Parse summary edits from reply"""
        edits = {}
        
        # Look for patterns like "Paper 3: [edited summary]"
        pattern = r'Paper (\d+):\s*(.+?)(?=Paper \d+:|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            paper_num = int(match[0])
            edited_text = match[1].strip()
            edits[paper_num] = edited_text
        
        return edits
    
    def _update_selected_papers(self, selections: List[int]):
        """Update selected status for papers"""
        # This would need to map paper numbers to IDs from the most recent email
        # Implementation depends on tracking which papers were in which email
        pass
    
    def _save_edits(self, edits: Dict[int, str]):
        """Save summary edits to database"""
        # This would need to map paper numbers to IDs and save edits
        # Implementation depends on tracking which papers were in which email
        pass

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class ArxivCurator:
    """Main orchestrator for the arxiv curation system"""
    
    def __init__(self):
        self.db = Database(Config.DB_PATH)
        self.arxiv_mgr = ArxivManager(self.db)
        self.claude_mgr = ClaudeManager()
        self.code_detector = CodeDetector()
        self.email_mgr = EmailManager(self.db)
    
    def run(self, check_replies_only: bool = False, force_run: bool = False):
        """Main execution flow"""
        logger.info("=" * 60)
        logger.info("Starting ArXiv Curator run")
        
        # Load topics configuration
        Config.load_topics()
        
        if check_replies_only:
            logger.info("Reply check only mode")
            self.email_mgr.check_replies()
            return
        
        if force_run:
            logger.info("Force run enabled - ignoring schedule restrictions")
        
        # Process each topic
        for topic_id, topic_config in Config.TOPICS.items():
            logger.info(f"\nProcessing topic: {topic_config['name']}")
            
            # Check if we should run this topic today
            if not force_run and not self._should_run_topic_today(topic_config):
                logger.info(f"Topic {topic_id} not scheduled to run today")
                continue
            
            # Phase 1: Collect new papers for this topic
            logger.info(f"Phase 1: Collecting papers for {topic_id}")
            new_papers = self.arxiv_mgr.collect_papers_for_topic(topic_id, topic_config)
            
            if not new_papers:
                logger.info(f"No new papers found for {topic_id}")
                continue
            
            # Phase 2: Filter papers with Claude
            logger.info(f"Phase 2: Filtering papers for {topic_id}")
            filtered_papers = self.claude_mgr.filter_papers(new_papers)
            
            # Phase 3: Detect code availability
            if Config.ENABLE_CODE_DETECTION:
                logger.info(f"Phase 3: Detecting code availability for {topic_id}")
                papers_with_code = self.code_detector.detect_code_in_papers(filtered_papers)
            else:
                papers_with_code = filtered_papers
            
            # Phase 4: Generate summaries
            logger.info(f"Phase 4: Generating summaries for {topic_id}")
            edit_examples = self.db.get_edit_examples()
            
            for paper in papers_with_code:
                summary = self.claude_mgr.generate_summary(paper, edit_examples)
                paper['original_summary'] = summary
                time.sleep(Config.REQUEST_DELAY_SECONDS)
            
            # Phase 5: Send email digest for this topic
            logger.info(f"Phase 5: Preparing email digest for {topic_id}")
            
            # Get topic-specific settings
            min_score = topic_config.get('min_relevance_score', Config.TOPIC_DEFAULTS['min_relevance_score'])
            max_papers = Config.MAX_PAPERS_PER_EMAIL
            recipient_email = topic_config.get('recipient_email', Config.DEFAULT_RECIPIENT_EMAIL)
            
            papers_to_send = self.db.get_unsent_papers(
                topic_id=topic_id,
                limit=max_papers,
                min_score=min_score
            )
            
            if papers_to_send:
                self.email_mgr.send_digest(papers_to_send, topic_id, topic_config['name'], recipient_email)
        
        # Phase 6: Check for replies (if enabled)
        if Config.ENABLE_REPLY_PROCESSING:
            logger.info("Phase 6: Checking for replies")
            self.email_mgr.check_replies()
        
        logger.info("ArXiv Curator run completed")
        logger.info("=" * 60)
    
    def _should_run_topic_today(self, topic_config: Dict[str, Any]) -> bool:
        """Check if we should run this topic today based on its schedule"""
        schedule_days = topic_config.get('schedule_days', Config.TOPIC_DEFAULTS['schedule_days'])
        
        if 'daily' in [d.lower() for d in schedule_days]:
            return True
        
        today = datetime.now().strftime('%a')
        return today in schedule_days
    

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ArXiv Paper Curator')
    parser.add_argument('--check-replies-only', action='store_true',
                       help='Only check for email replies')
    parser.add_argument('--force-run', action='store_true',
                       help='Run regardless of day restrictions')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without sending emails')
    parser.add_argument('--rebuild-summaries', action='store_true',
                       help='Regenerate all summaries')
    parser.add_argument('--retry-emails', action='store_true',
                       help='Retry sending failed emails from queue')
    
    args = parser.parse_args()
    
    # Override config with command line args
    if args.dry_run:
        Config.DRY_RUN = True
    
    try:
        curator = ArxivCurator()
        
        if args.retry_emails:
            logger.info("Retrying failed emails")
            curator.email_mgr.send_pending_emails()
        elif args.rebuild_summaries:
            logger.info("Rebuilding all summaries")
            # Implementation for rebuilding summaries
            pass
        else:
            curator.run(check_replies_only=args.check_replies_only, force_run=args.force_run)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()