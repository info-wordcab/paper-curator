# ArXiv Paper Curator

An automated system for discovering, filtering, and summarizing relevant research papers from arXiv with continuous learning from user feedback. Supports multiple research topics with individual configurations and scheduling.

## Features

- **Multi-Topic Support**: Configure multiple research topics with different search criteria and schedules
- **Advanced Search**: Supports arXiv Advanced Search URLs for complex queries
- **AI-Powered Filtering**: Uses Claude Opus to evaluate paper relevance (1-10 scale)
- **Code Detection**: Automatically identifies papers with GitHub repositories
- **Smart Summarization**: Generates concise summaries using Claude with feedback learning
- **Email Digests**: Sends HTML/text digests via Postmark with customizable scheduling
- **Feedback Learning**: Learns from email reply edits to improve future summaries
- **Database Tracking**: SQLite database for papers, edits, and email queue management
- **Retry Capability**: Email queue system with automatic retry for failed sends

## Setup

1. **Clone and Install**:
   ```bash
   git clone <repository>
   cd arxiv-curator
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   - Copy `.env.example` to `.env`
   - Add your API keys:
     - `CLAUDE_API_KEY`: Claude API key from Anthropic
     - `POSTMARK_SERVER_TOKEN`: Postmark API token
     - `POSTMARK_FROM_EMAIL`: Your verified sender email
     - `DEFAULT_RECIPIENT_EMAIL`: Default recipient for digests

3. **Configure Topics**:
   - Edit `topics.yaml` to define your research interests
   - Each topic can have:
     - Multiple search queries or Advanced Search URLs
     - Custom schedule (days of week)
     - Minimum relevance score threshold
     - Recipient email override

4. **Set Up Gmail** (optional, for reply processing):
   ```python
   import ezgmail
   ezgmail.init()  # Follow OAuth flow
   ```
   This creates `credentials.json` and `token.json` files.

5. **Test Run**:
   ```bash
   python arxiv_curator.py --dry-run --force-run
   ```

6. **Schedule with Cron**:
   ```bash
   crontab -e
   # Add the cron jobs from crontab_config.txt
   ```

## Usage

### Command Line Options

- `python arxiv_curator.py` - Normal run (respects day schedule)
- `python arxiv_curator.py --force-run` - Run regardless of schedule
- `python arxiv_curator.py --check-replies-only` - Only process email replies
- `python arxiv_curator.py --dry-run` - Test without sending emails
- `python arxiv_curator.py --rebuild-summaries` - Regenerate all summaries
- `python arxiv_curator.py --retry-emails` - Retry failed emails from queue

### Email Feedback Format

Reply to digest emails with:
```
Selection: 1-3, 5, 7-10, 15

Edits:
Paper 3: [Your improved summary here]
Paper 7: [Another improved summary]
```

## Configuration

### Topics Configuration (`topics.yaml`)

```yaml
topics:
  topic_id:
    name: "Topic Display Name"
    searches:
      - query: "keyword search"  # Simple search
      - url: "https://arxiv.org/search/advanced?..."  # Advanced search URL
    max_results: 30
    days_back: 7
    min_relevance_score: 7
    schedule_days: ["Mon", "Wed", "Fri"]  # or ["daily"]
    recipient_email: "optional@override.com"
```

### Environment Variables (`.env`)

Key settings:
- `CLAUDE_MODEL`: Model to use (default: claude-opus-4-20250514)
- `MAX_PAPERS_PER_EMAIL`: Papers per digest (default: 20)
- `MIN_PAPERS_TO_SEND`: Minimum papers required to send email (default: 3)
- `ENABLE_CODE_DETECTION`: Detect GitHub links (default: true)
- `ENABLE_PDF_DOWNLOAD`: Download PDFs for analysis (default: true)
- `ENABLE_REPLY_PROCESSING`: Process email replies (default: true)

## Database Schema

The system uses SQLite with four main tables:
- `papers`: All discovered papers with metadata, scores, and summaries
- `summary_edits`: User feedback for continuous improvement
- `email_logs`: Tracking of sent digests and replies
- `email_queue`: Queue for reliable email delivery with retry

## Monitoring

Check logs in `./logs/arxiv_curator.log` for:
- Papers discovered
- Filtering decisions
- Email status
- Error messages

## Troubleshooting

**No papers found**: 
- Check search terms are not too specific
- Verify arXiv API is accessible
- For Advanced Search URLs, test them in browser first
- Check `days_back` setting isn't too restrictive

**Emails not sending**:
- Verify Postmark token and sender email
- Check recipient email is correct
- Use `--retry-emails` to retry failed sends
- Check email queue in database

**Replies not processing**:
- Ensure ezgmail is initialized
- Check Gmail API credentials
- Verify `ENABLE_REPLY_PROCESSING` is true

**High API costs**:
- Adjust `BATCH_SIZE` for filtering
- Increase `MIN_RELEVANCE_SCORE` threshold
- Reduce `max_results` per topic

## Architecture

The system follows a modular design:
- `ArxivManager`: Handles arXiv API searches and Advanced URL parsing
- `ClaudeManager`: Manages AI filtering and summarization
- `CodeDetector`: Identifies GitHub repositories in papers
- `EmailManager`: Handles Postmark sending and Gmail reply processing
- `Database`: SQLite interface with migration support

## Contributing

The system is designed to be extensible. Key extension points:
- Add new paper sources beyond arXiv
- Implement different AI models for summarization
- Add support for other email providers
- Enhance code detection algorithms
- Add web interface for configuration

## License

MIT License - See LICENSE file for details