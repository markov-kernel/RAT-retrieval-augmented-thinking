# RAT (Retrieval Augmented Thinking) Documentation

## Project Overview
RAT (Retrieval Augmented Thinking) is a system designed to enhance research and information processing through a multi-agent architecture. The system uses various specialized agents to explore, search, reason about, and execute tasks based on retrieved information.

## Project Structure

```
rat/
├── research/                    # Core research functionality
│   ├── agents/                 # Agent implementations
│   │   ├── base.py            # Base agent class and common functionality
│   │   ├── context.py         # Context management agent
│   │   ├── execute.py         # Task execution agent
│   │   ├── explore.py         # Information exploration agent
│   │   ├── reason.py          # Reasoning and analysis agent
│   │   └── search.py          # Information search agent
│   ├── orchestrator.py         # Main orchestration logic
│   ├── output_manager.py       # Manages research outputs
│   ├── perplexity_client.py    # Perplexity API integration
│   └── firecrawl_client.py     # Firecrawl service integration
```

## Core Components

### 1. Agent System

The system is built on a multi-agent architecture where each agent specializes in a specific aspect of the research process:

#### Base Agent (`agents/base.py`)
- Provides foundational agent functionality
- Implements common methods and utilities used by all agents
- Handles basic agent communication and state management

#### Context Agent (`agents/context.py`)
- Manages research context and knowledge state
- Maintains and updates the contextual information
- Helps in maintaining coherence across research sessions

#### Execute Agent (`agents/execute.py`)
- Handles task execution and action implementation
- Converts high-level plans into concrete actions
- Manages the execution flow of research tasks

#### Explore Agent (`agents/explore.py`)
- Conducts exploratory research
- Identifies new areas of investigation
- Generates initial insights and directions

#### Reason Agent (`agents/reason.py`)
- Performs analytical reasoning on gathered information
- Draws conclusions and makes recommendations
- Synthesizes information from multiple sources

#### Search Agent (`agents/search.py`)
- Handles information retrieval operations
- Implements search strategies
- Manages query formulation and result processing

### 2. Orchestration (`orchestrator.py`)
- Coordinates the activities of all agents
- Manages the overall research workflow
- Handles agent communication and task distribution
- Ensures coherent research progression

### 3. Output Management (`output_manager.py`)
- Manages research outputs and artifacts
- Handles formatting and organization of results
- Maintains research history and documentation

### 4. External Integrations

The system integrates with external services through specialized clients:

#### 1. Perplexity Integration (`perplexity_client.py`)

##### Configuration
```python
class PerplexityClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai"
        )
        self.model = "sonar-pro"
```

##### Features
1. Intelligent Search
   - Advanced query processing
   - Source citation extraction
   - URL validation

2. Response Processing
   ```python
   {
       "content": str,  # Search results with citations
       "urls": List[str]  # Extracted source URLs
   }
   ```

3. URL Management
   - Citation format parsing: `[Source: URL]`
   - URL validation and accessibility checks
   - Duplicate URL removal

#### 2. Firecrawl Integration (`firecrawl_client.py`)

##### Configuration
```python
class FirecrawlClient:
    def __init__(self):
        self.api_key = os.getenv("FIRECRAWL_API_KEY")
        self.app = FirecrawlApp(api_key=self.api_key)
```

##### Features
1. Content Extraction
   ```python
   def extract_content(self, url: str) -> Dict[str, Any]:
       # Scrapes webpage content
       # Processes and cleans extracted data
       # Returns structured content
   ```

2. Content Processing
   - Markdown formatting
   - Text cleaning
   - Metadata extraction

3. Response Structure
   ```python
   {
       "title": str,
       "text": str,
       "metadata": {
           "url": str,
           "author": str,
           "published_date": str,
           "domain": str,
           "word_count": int,
           "language": str,
           "status_code": int
       }
   }
   ```

### Integration Best Practices

#### 1. Error Handling
- Graceful failure handling
- Detailed error reporting
- Fallback mechanisms

#### 2. Rate Limiting
- Respect API limits
- Implement backoff strategies
- Queue long-running requests

#### 3. Data Validation
- URL validation
- Content format verification
- Metadata completeness checks

#### 4. Security
- API key management
- URL sanitization
- Content safety checks

### Environment Configuration

Required environment variables:
```
PERPLEXITY_API_KEY=your_perplexity_key
FIRECRAWL_API_KEY=your_firecrawl_key
```

### Integration Flow

1. Research Initiation
   - Perplexity search for initial sources
   - URL extraction and validation

2. Content Gathering
   - Firecrawl extraction of validated URLs
   - Content processing and cleaning

3. Data Integration
   - Merge search and crawl results
   - Update research context
   - Track source citations

## Detailed Implementation

### Agent System Architecture

The agent system is built on a robust foundation defined in `base.py`, which provides:

#### 1. Decision Making Framework
```python
class DecisionType(Enum):
    SEARCH = "search"    # New search query needed
    EXPLORE = "explore"  # URL exploration needed
    REASON = "reason"    # Deep analysis needed
    EXECUTE = "execute"  # Code execution needed
    TERMINATE = "terminate"  # Research complete
```

#### 2. Research Decisions
Each agent makes decisions represented by the `ResearchDecision` class:
- `decision_type`: Type of action recommended
- `priority`: Priority level (0-1)
- `context`: Additional parameters
- `rationale`: Explanation for the decision

#### 3. Base Agent Interface
All agents inherit from `BaseAgent` which defines:
- Abstract methods for analysis and execution
- Decision logging and metrics tracking
- Configuration management
- History tracking

### Orchestration System

The `ResearchOrchestrator` class (`orchestrator.py`) manages the research workflow:

#### 1. Initialization
- Sets up all agent instances
- Configures external clients (Perplexity, Firecrawl)
- Initializes output management
- Sets research parameters (iterations, confidence thresholds)

#### 2. Research Process
```python
def start_research(self, question: str) -> Dict[str, Any]:
    # Creates research directory
    # Initializes research context
    # Runs iterations until completion
    # Generates and saves final output
```

#### 3. Iteration Workflow
Each research iteration:
1. Gathers decisions from all agents
2. Sorts decisions by priority
3. Executes decisions through appropriate agents
4. Updates research context with results
5. Tracks metrics and progress

#### 4. Decision Execution
```python
def _run_iteration(self, iteration_number: int) -> ResearchIteration:
    # Gathers agent decisions
    # Sorts by priority
    # Executes through appropriate agents
    # Updates context
    # Tracks metrics
```

### External Integrations

#### 1. Perplexity Integration
- Handles advanced research queries
- Manages API communication
- Processes structured responses

#### 2. Firecrawl Integration
- Manages web crawling
- Extracts relevant content
- Processes web resources

### Data Structures

#### 1. Research Context
Maintains the state of research:
- Initial question
- Gathered content
- Current findings
- Research progress

#### 2. Content Management
```python
@dataclass
class ContentItem:
    content: Any
    content_type: ContentType
    priority: float
    metadata: Dict[str, Any]
```

#### 3. Research Iteration
```python
@dataclass
class ResearchIteration:
    iteration_number: int
    decisions_made: List[ResearchDecision]
    content_added: List[ContentItem]
    metrics: Dict[str, Any]
    timestamp: float
```

### Metrics and Monitoring

The system tracks various metrics:
1. Agent Performance
   - Decisions made
   - Successful executions
   - Failed executions
   - Execution time

2. Research Progress
   - Iteration metrics
   - Content gathered
   - Decision success rates
   - Overall completion time

3. Quality Metrics
   - Content relevance
   - Decision confidence
   - Research coverage

## Workflow

1. The system typically begins with the Orchestrator initiating a research task
2. The Context Agent establishes the research context
3. The Explore Agent conducts initial exploration
4. The Search Agent retrieves relevant information
5. The Reason Agent analyzes the gathered information
6. The Execute Agent implements any required actions
7. The Output Manager organizes and stores the results

## Configuration

The system uses various configuration files:
- `.env`: Environment variables and API keys
- `pyproject.toml`: Project dependencies and metadata
- `requirements.txt`: Python package dependencies

## Development Guidelines

1. Follow SOLID principles and maintain modular architecture
2. Keep functions and classes focused and single-purpose
3. Implement proper error handling and logging
4. Use configuration files for environment-specific details
5. Maintain comprehensive documentation
6. Follow the established code review process

## Future Improvements

- [ ] Enhance agent communication patterns
- [ ] Implement additional specialized agents
- [ ] Improve error handling and recovery
- [ ] Add more external integrations
- [ ] Enhance output formatting options
- [ ] Implement advanced caching mechanisms 

### Output Management System

The `OutputManager` class (`output_manager.py`) provides a comprehensive system for managing research outputs:

#### 1. Directory Structure
```
research_outputs/
├── {timestamp}_{question}/     # Research session directory
│   ├── research_paper.md      # Final research paper
│   ├── research_info.json     # Paper metadata
│   ├── metadata.json          # Session metadata
│   ├── iteration_metrics.json # Performance metrics
│   └── states/               # Context state history
│       └── context_state_{timestamp}.json
```

#### 2. Research Session Management
```python
def create_research_dir(self, question: str) -> Path:
    # Creates timestamped directory
    # Initializes metadata
    # Returns directory path
```

#### 3. Output Types

##### Research Paper
- Markdown format for content
- JSON metadata including:
  - Title
  - Sources
  - Performance metrics
  - Completion status

##### Context States
- Periodic snapshots of research progress
- Limited to 5 most recent states
- Includes:
  - Current findings
  - Active decisions
  - Research progress

##### Metrics Tracking
The system maintains detailed metrics at multiple levels:

1. Session Level
   - Start/end times
   - Overall status
   - Aggregate metrics

2. Iteration Level
   - Decisions made
   - New content gathered
   - Time per iteration
   - Success rates

3. Summary Metrics
```python
{
    "total_iterations": int,
    "total_decisions": int,
    "total_new_content": int,
    "total_time": float,
    "avg_decisions_per_iteration": float,
    "avg_new_content_per_iteration": float,
    "avg_time_per_iteration": float
}
```

#### 4. File Management
- Automatic directory creation
- Safe filename sanitization
- State cleanup for space efficiency
- Atomic updates for metadata

### Integration Points

#### 1. With Orchestrator
- Creates research directories
- Saves iteration results
- Updates session metadata
- Generates final output

#### 2. With Agents
- Records agent decisions
- Tracks performance metrics
- Stores intermediate results

#### 3. With External Systems
- Manages API response storage
- Handles web crawl results
- Saves processed content

### Best Practices

1. File Organization
   - Use timestamped directories
   - Maintain clear hierarchy
   - Clean up old states

2. Metadata Management
   - Track research progress
   - Store performance metrics
   - Maintain session state

3. Error Handling
   - Safe file operations
   - Atomic updates
   - Backup state management 
