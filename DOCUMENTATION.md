# RAT Documentation

## Overview

RAT (Retrieval Augmented Thinking) is a multi-agent research system that enhances AI responses through structured reasoning and research. The system uses specialized agents working together to gather, analyze, and synthesize information.

## Core Components

### 1. Agent Architecture

#### Search Agent (`agents/search.py`)
- Handles information retrieval operations using Perplexity API
- Implements search strategies and query formulation
- Manages result processing and URL extraction

#### Explore Agent (`agents/explore.py`)
- Manages URL content extraction using Firecrawl API
- Handles webpage scraping and content cleaning
- Supports batch processing and rate limiting

#### Reasoning Agent (`agents/reason.py`)
- Uses DeepSeek Reasoner for primary content analysis
- Uses GPT-4o-mini for JSON repair when needed
- Manages research flow and decision making
- Supports parallel processing and content chunking

### 2. Orchestration (`orchestrator.py`)
- Coordinates agent activities
- Manages research workflow
- Handles agent communication
- Ensures coherent research progression

### 3. Output Management (`output_manager.py`)
- Manages research outputs and artifacts
- Handles formatting and organization
- Maintains research history

## Environment Configuration

Required environment variables:
```
PERPLEXITY_API_KEY=your_perplexity_key
FIRECRAWL_API_KEY=your_firecrawl_key
DEEPSEEK_API_KEY=your_deepseek_key
GPT4OMINI_API_KEY=your_gpt4omini_key
```

## Integration Flow

1. Research Initiation
   - User submits research question
   - System initializes research context

2. Content Gathering
   - Search Agent queries Perplexity API
   - Explore Agent extracts content via Firecrawl
   - Content is processed and cleaned

3. Analysis
   - Reasoning Agent analyzes content using DeepSeek
   - JSON outputs are validated and repaired if needed using GPT-4o-mini
   - System makes decisions about next steps

4. Output Generation
   - Results are synthesized into a research paper
   - Sources are tracked and cited
   - Metrics are collected and reported

## Best Practices

### 1. Error Handling
- Graceful failure handling
- Detailed error reporting
- Fallback mechanisms

### 2. Rate Limiting
- Respect API limits
- Implement backoff strategies
- Queue long-running requests

### 3. Data Validation
- URL validation
- Content format verification
- Metadata completeness checks

### 4. Security
- API key management
- URL sanitization
- Content safety checks

## Development Guidelines

1. Follow SOLID principles
2. Keep functions focused and single-purpose
3. Implement proper error handling
4. Use configuration files for environment-specific details
5. Maintain comprehensive documentation
6. Follow the established code review process

## Future Improvements

1. Enhanced error handling and recovery
2. Improved content deduplication
3. More sophisticated research strategies
4. Better handling of rate limits
5. Enhanced metrics and monitoring
