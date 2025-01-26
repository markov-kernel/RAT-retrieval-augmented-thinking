# RAT (Retrieval Augmented Thinking) - Functional Design

## System Overview
RAT is a multi-agent research system that enhances AI responses through structured reasoning and research. The system uses multiple specialized agents working together to gather, analyze, and synthesize information.

## Core Components

### 1. Agent Architecture
- **SearchAgent**: Uses Perplexity API for intelligent web searches
- **ExploreAgent**: Uses Firecrawl API for content extraction from URLs
- **ReasoningAgent**: Uses DeepSeek for analysis and GPT-4o-mini for JSON repair
- **ResearchOrchestrator**: Coordinates agent interactions and research flow

### 2. Key Interfaces
- **PerplexityClient**: Interface for web search functionality
- **FirecrawlClient**: Interface for web scraping and content extraction
- **OutputManager**: Handles research output and metrics storage

### 3. Data Flow
1. User submits research question
2. SearchAgent generates and executes search queries
3. ExploreAgent extracts content from discovered URLs
4. ReasoningAgent analyzes gathered information
5. System generates final research paper

## Technical Decisions

### API Integration
- Perplexity API for intelligent search
- Firecrawl API for web scraping
- DeepSeek API for reasoning
- GPT-4o-mini for JSON repair

### Environment Configuration
Required environment variables:
- PERPLEXITY_API_KEY
- FIRECRAWL_API_KEY
- DEEPSEEK_API_KEY
- GPT4OMINI_API_KEY

### Parallel Processing
- Supports parallel execution of agent decisions
- Rate limiting and concurrency controls
- Thread-safe operations for shared resources

## Future Improvements
1. Enhanced error handling and recovery
2. Improved content deduplication
3. More sophisticated research strategies
4. Better handling of rate limits
5. Enhanced metrics and monitoring

## Security Considerations
- API keys stored in environment variables
- No hardcoded credentials
- Safe URL handling and validation
- Input sanitization

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

# RAT (Retrieval Augmented Thinking) - Parallel Processing Implementation

## Overview
This document outlines the implementation of parallel processing capabilities in the RAT system to improve performance and efficiency.

## Key Components

### 1. Orchestrator
- Manages parallel execution of tasks within each iteration
- Groups decisions by type (SEARCH, EXPLORE, REASON, EXECUTE)
- Enforces concurrency limits per task type
- Uses ThreadPoolExecutor for parallel task execution
- Maintains iteration state and context

### 2. ReasoningAgent
- Controls research flow and decision making
- Produces multiple parallel decisions when appropriate
- Analyzes results from parallel tasks
- Determines when to terminate research

### 3. Parallel Processing Features
- Concurrent searches with PerplexityClient
- Parallel web exploration with FirecrawlClient
- Distributed reasoning tasks with DeepSeek
- Concurrent execution tasks where applicable

### 4. Concurrency Limits
- Search tasks: 10 concurrent
- Explore tasks: 10 concurrent
- Reason tasks: 5 concurrent
- Execute tasks: 5 concurrent

## Implementation Plan

1. Update base agent infrastructure
   - Add parallel processing support
   - Implement concurrency controls
   - Add metrics tracking

2. Enhance Orchestrator
   - Add parallel task execution
   - Implement decision grouping
   - Add batch processing
   - Update context management

3. Modify ReasoningAgent
   - Support multiple parallel decisions
   - Add parallel analysis capabilities
   - Implement termination logic

4. Update Supporting Agents
   - Add parallel search capabilities
   - Implement concurrent exploration
   - Enable parallel execution

## Technical Considerations

### Threading Model
- Use ThreadPoolExecutor for I/O bound tasks
- Maintain thread safety in context updates
- Handle exceptions in parallel tasks

### Rate Limiting
- Implement per-service rate limiting
- Track API usage across parallel tasks
- Handle service throttling gracefully

### Error Handling
- Graceful degradation on partial failures
- Retry logic for failed tasks
- Comprehensive error reporting

## Future Improvements

### TODO
1. Implement asyncio for true asynchronous operations
2. Add dynamic concurrency adjustment based on system load
3. Implement caching for parallel task results
4. Add distributed processing capabilities
5. Implement advanced rate limiting strategies 