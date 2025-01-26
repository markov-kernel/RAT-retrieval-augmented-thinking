# RAT - Retrieval Augmented Thinking
## Functional Design Document

### Overview
RAT (Retrieval Augmented Thinking) is a multi-agent research system that leverages various AI models to conduct comprehensive research and analysis. The system uses specialized agents for different aspects of the research process.

### Core Components

#### 1. Main Components
- Research Orchestrator: Coordinates all agents and manages the research flow
- Output Manager: Handles research output formatting and storage
- Logging System: Centralized logging with file-based storage

#### 2. Specialized Agents
- Search Agent (Perplexity): Conducts intelligent web searches
- Explore Agent (Firecrawl): Extracts content from URLs
- Reasoning Agent (DeepSeek): Analyzes content and makes decisions
- Execution Agent (Claude): Generates code and structured output

### Key Requirements

1. Architecture and Design
- Follow SOLID principles and modular architecture
- Implement clear separation of concerns
- Use appropriate design patterns
- Maintain clean code structure

2. Error Handling and Logging
- Centralized logging in rat.log (overwritten each run)
- Consistent error handling across all components
- Proper exception handling with meaningful messages
- Log levels: INFO for normal operations, ERROR for issues

3. API Integration
- Remove invalid "role": "system" usage from API calls
- Use single prompt format for all AI model interactions
- Handle API rate limits and retries
- Proper error handling for API failures

4. Performance and Scalability
- Support parallel processing where appropriate
- Implement rate limiting for API calls
- Handle large content chunks efficiently
- Manage memory usage for large datasets

### Technical Specifications

1. Logging Configuration
- Location: rat.log in root directory
- Format: "%(asctime)s %(levelname)s %(name)s %(message)s"
- Mode: Overwrite on each run
- Level: INFO as default

2. API Interactions
- Perplexity: Single prompt format for searches
- DeepSeek: Single prompt for content analysis
- Claude: Single prompt for code generation
- Proper API key management via environment variables

3. Code Structure
- Modular design with clear responsibilities
- Consistent error handling patterns
- Clean separation of concerns
- Well-documented interfaces

### Future Improvements (TODOs)
1. Add more sophisticated error recovery mechanisms
2. Implement caching for API responses
3. Add metrics collection and monitoring
4. Enhance parallel processing capabilities
5. Add support for more AI models/providers

### Non-Functional Requirements
1. Performance
- Response time < 5s for basic operations
- Graceful handling of API timeouts
- Efficient memory usage

2. Security
- Secure API key management
- No sensitive data in logs
- Input validation and sanitization

3. Maintainability
- Clear documentation
- Consistent coding style
- Comprehensive logging
- Easy configuration management

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