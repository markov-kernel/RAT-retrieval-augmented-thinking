# RAT Research - Functional Design Document

## Overview
RAT Research is an enhanced version of the Retrieval Augmented Thinking (RAT) system that adds web research capabilities through Perplexity API and Jina Reader for comprehensive research paper generation.

## Core Components

### 1. Model Chain (from RAT)
- Maintains the existing DeepSeek reasoning capabilities
- Integrates with OpenRouter for response generation
- Preserves conversation history and model switching functionality

### 2. Research Components (New)
- Perplexity API Integration for web searches
- Jina Reader API for web scraping and content extraction
- URL extraction and validation
- Content aggregation and synthesis

### 3. Output Generation
- Markdown formatted research papers
- Structured sections (Introduction, Methods, Findings, Conclusion)
- Citation management
- Source tracking

## Workflow

1. User Input Processing
   - Accept research questions via command line interface
   - Parse and validate input

2. Initial Research Phase
   - Conduct Perplexity API searches
   - Extract relevant URLs from search results
   - Store initial findings

3. Deep Research Phase
   - Use Jina Reader to scrape and analyze cited URLs
   - Extract relevant content and metadata
   - Validate and clean extracted data

4. Synthesis Phase
   - Use DeepSeek for reasoning about collected information
   - Generate structured research outline
   - Synthesize findings into coherent sections

5. Output Generation
   - Format research paper in Markdown
   - Include proper citations and references
   - Generate executive summary

## Technical Requirements

### Dependencies
- OpenAI API (for DeepSeek and OpenRouter)
- Perplexity API
- Jina Reader API
- Web scraping utilities
- Markdown processing libraries

### Environment Configuration
- API keys stored in .env file
- Configuration management
- Rate limiting and error handling

### Data Structures
- Research session object
- Source tracking
- Citation management
- Content cache

## Command Line Interface

### Commands
- `research <question>` - Start new research
- `model <name>` - Switch models (inherited from RAT)
- `reasoning` - Toggle reasoning visibility
- `clear` - Clear research session
- `export` - Export research paper
- `quit` - Exit application

## Future Improvements
- PDF export capability
- Multiple research paper formats
- Citation style options
- Research progress tracking
- Source reliability scoring
- Content summarization options

## Error Handling
- API failure recovery
- Rate limit management
- Invalid URL handling
- Content extraction fallbacks
- Session persistence

## Security Considerations
- API key protection
- URL validation
- Content sanitization
- Rate limiting
- Error logging 