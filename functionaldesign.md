# RAT (Retrieval Augmented Thinking) - Functional Design

## Overview
RAT is a multi-agent system for conducting research and generating insights through a combination of specialized AI agents. The system uses multiple AI models and APIs to search, explore, reason about, and execute tasks based on user queries.

## System Architecture

### Core Components

1. **Search Agent (Perplexity)**
   - Handles web search queries
   - Returns relevant content and URLs
   - Uses Perplexity API for intelligent search

2. **Explore Agent (Firecrawl)**
   - Extracts content from discovered URLs
   - Processes and cleans webpage content
   - Uses Firecrawl API for content extraction

3. **Reasoning Agent (DeepSeek)**
   - Analyzes gathered information
   - Supports parallel processing for large contexts
   - Uses deepseek-reasoner for analysis

4. **Execution Agent (Claude)**
   - Generates code and structured output
   - Handles specific execution tasks
   - Uses claude-3-5-sonnet for precise outputs

### Orchestrator
- Coordinates agent interactions
- Manages research workflow
- Handles data persistence and retrieval

## Data Flow

1. User submits research query
2. Search Agent gathers initial information
3. Explore Agent processes discovered URLs
4. Reasoning Agent analyzes collected data
5. Optional: Execution Agent generates structured output
6. Results saved to research_outputs directory

## Key Features

- Multi-agent architecture
- Iterative research process
- Parallel processing capability
- Structured output generation
- Persistent storage of results

## Technical Requirements

### Environment Variables
```
DEEPSEEK_API_KEY
PERPLEXITY_API_KEY
FIRECRAWL_API_KEY
ANTHROPIC_API_KEY
```

### Dependencies
- openai
- python-dotenv
- rich
- prompt_toolkit
- requests
- firecrawl
- anthropic

## Output Format

Research outputs are stored in timestamped directories containing:
- research_paper.md (main findings)
- metadata.json (research metadata)
- sources/ (extracted source content)

## Future Improvements

1. Enhanced parallelization
2. Additional agent types
3. Improved source validation
4. Advanced error handling
5. Query optimization

## Usage

```bash
# Install
pip install -e .

# Run
python rat_agentic.py
# or
rat-agentic
``` 