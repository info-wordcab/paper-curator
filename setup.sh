#!/bin/bash

echo "ArXiv Paper Curator Setup"
echo "========================"

# Create directory structure
echo "Creating directories..."
mkdir -p logs
mkdir -p pdfs
mkdir -p temp
mkdir -p backups
mkdir -p prompts

# Copy prompt templates
echo "Setting up prompt templates..."
# The prompts are defined above - copy them to the prompts/ directory

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Initialize ezgmail (requires manual OAuth setup)
echo ""
echo "Setting up Gmail access..."
echo "You need to:"
echo "1. Go to https://console.cloud.google.com/"
echo "2. Create a new project or select existing"
echo "3. Enable Gmail API"
echo "4. Create OAuth 2.0 credentials"
echo "5. Download credentials.json to this directory"
echo "6. Run: python3 -c 'import ezgmail; ezgmail.init()'"
echo ""

# Database initialization happens automatically on first run

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "Please edit .env and add your API keys and configuration"
    cp .env.example .env
fi

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys and settings"
echo "2. Set up Gmail OAuth (see instructions above)"
echo "3. Test with: python3 arxiv_curator.py --dry-run"
echo "4. Add to crontab for automated running"
