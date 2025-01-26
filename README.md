# RAT - Retrieval Augmented Thinking

A multi-agent research system that leverages AI to perform comprehensive research tasks. The system consists of four specialized agents working together to gather, analyze, and process information:

1. **Search Agent**: Utilizes Perplexity for intelligent web searching
2. **Explore Agent**: Handles URL content extraction using Firecrawl
3. **Reasoning Agent**: Analyzes content using DeepSeek
4. **Execution Agent**: Generates code and structured output using Claude

## Features

- Coordinated multi-agent research workflow
- Intelligent web searching and content extraction
- Advanced content analysis and reasoning
- Code generation and structured output formatting
- Command-line interface for easy interaction

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows

# Install package in development mode
pip install -e .
```

## Usage

```bash
# Run the research system
python -m rat.research.main
```

## Configuration

The system can be configured through environment variables or a configuration file. Key settings include:

- API keys for various services (Perplexity, Firecrawl, DeepSeek, Claude)
- Research iteration limits
- Content thresholds
- Agent-specific parameters

## License

MIT License 