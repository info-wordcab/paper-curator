# ArXiv Paper Curator

An automated system for discovering, filtering, and summarizing relevant research papers from arXiv with continuous learning from user feedback.

## Features

- **Automated Paper Discovery**: Searches arXiv for papers matching your research interests
- **AI-Powered Filtering**: Uses Claude to evaluate paper relevance and quality
- **Code Detection**: Identifies papers with available implementations
- **Smart Summarization**: Generates concise, informative summaries that improve over time
- **Email Digests**: Sends formatted digests of the best papers
- **Feedback Learning**: Learns from your edits to improve future summaries
- **Database Tracking**: Maintains history of all papers and interactions

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
     - Claude API key from Anthropic
     - Postmark server token
     - Configure email addresses

3. **Set Up Gmail** (for reply processing):
   ```python
   import ezgmail
   ezgmail.init()  # Follow OAuth flow
   ```

4. **Test Run**:
   ```bash
   python arxiv_curator.py --dry-run
   ```

5. **Schedule with Cron**:
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

### Email Feedback Format

Reply to digest emails with:
```
Selection: 1-3, 5, 7-10, 15

Edits:
Paper 3: [Your improved summary here]
Paper 7: [Another improved summary]
```

## Configuration

Key settings in `.env`:

- `ARXIV_SEARCH_TERMS`: Your research interests
- `ARXIV_SEARCH_DAYS`: When to run (Mon,Wed,Fri or daily)
- `MAX_PAPERS_PER_EMAIL`: Papers per digest (default: 20)
- `MIN_RELEVANCE_SCORE`: Quality threshold (1-10 scale)

## Database Schema

The system maintains three main tables:
- `papers`: All discovered papers with metadata
- `summary_edits`: Your feedback for continuous improvement
- `email_logs`: Tracking of sent digests

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

**Emails not sending**:
- Verify Postmark token and sender email
- Check recipient email is correct

**Replies not processing**:
- Ensure ezgmail is initialized
- Check Gmail API credentials

## Contributing

The system is designed to be extensible. Key extension points:
- Add new paper sources beyond arXiv
- Implement different summarization strategies
- Add support for other email providers
- Enhance code detection algorithms

## License

MIT License - See LICENSE file for details