# RAT (Retrieval Augmented Thinking)

**A multi-agent research system integrating web search, content exploration, and advanced reasoning capabilities (via Gemini).**

RAT uses a coordinated set of specialized agents for retrieving, exploring, and analyzing data, producing comprehensive AI-driven research outputs.

## Key Functionalities

### 1. Search
- Leverages the **Perplexity** API (via `PerplexityClient`) to perform targeted web searches, track queries, and gather relevant results.

### 2. URL Exploration
- Uses the **Firecrawl** client (`FirecrawlClient`) to scrape and extract content from web pages, returning structured information.

### 3. Advanced Reasoning
- Employs a **Gemini 2.0 Flash Thinking** model (via `ReasoningAgent`) to analyze aggregated content, detect knowledge gaps, and produce in-depth analyses.

### 4. Orchestration
- The **ResearchOrchestrator** synchronizes agents, coordinates decisions, and manages research iterations end-to-end.

### 5. Output Management
- Generates a consolidated Markdown research paper and can export it to a PDF using WeasyPrint.
- Organizes and saves intermediate results, logs, and final outputs for reproducibility.

## Use Cases

### 1. In-Depth Topic Research
- Investigate an unfamiliar subject by running broad searches, exploring specific URLs, and generating summarized insights and references.

### 2. Competitive Analysis
- Compare features, pricing, or other aspects of multiple competitors, synthesizing content from various sources in one final report.

### 3. Academic Literature Review
- Conduct structured searches, collect summaries from relevant materials, and produce a cohesive paper or knowledge summary.

### 4. Corporate Intelligence
- Continuously track references, gather specialized data, and provide reasoned analytics on markets, companies, or technologies.

## Project Overview

RAT is designed to automate the gathering, analysis, and summarization of information from the web:

### Multi-Agent System
Each agent (Search, Explore, Reasoning) has distinct responsibilities, making decisions and producing content in parallel when needed.

### ResearchContext
A shared data model that stores the research question, branch-based content (search results, explored content, analysis), and token counts.

### Decision Flow
1. **ReasoningAgent** proposes next steps (search, explore, or finalize)
2. **SearchAgent** executes web queries when directed
3. **ExploreAgent** extracts content from identified URLs
4. The **ReasoningAgent** analyzes newly added content, identifies gaps, and refines the process until research is complete

### ResearchOrchestrator
Oversees the main loop, orchestrating agent interactions over multiple iterations. It terminates once the system decides no further research steps are needed or a termination decision is reached.

### OutputManager
Handles file structure, saving intermediate states (e.g., logs, partial results, analysis outputs), final Markdown, and PDF generation.

## Installation & Requirements

### 1. Python Version
- Requires **Python 3.7** or higher

### 2. Dependencies
Listed in `setup.py`. Major libraries include:
- `openai`
- `python-dotenv`
- `rich`
- `prompt_toolkit`
- `requests`
- `weasyprint` (for PDF generation)

Some classes also reference `google.generativeai` and external services, so you may need to install or configure these separately.

### 3. Environment Variables
Create a `.env` file or export variables for third-party keys:
```env
PERPLEXITY_API_KEY="your_key_here"
FIRECRAWL_API_KEY="your_key_here"
GEMINI_API_KEY="your_key_here"
```

### 4. Installation Steps
```bash
git clone https://github.com/Doriandarko/RAT-retrieval-augmented-thinking.git
cd RAT-retrieval-augmented-thinking
pip install .
```

Or install in editable mode for development:
```bash
pip install -e .
```

## Repository Structure

### Primary Directories & Files

```
rat/
├── __init__.py
└── research/
    ├── __init__.py
    ├── __main__.py
    ├── agents/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── context.py
    │   ├── explore.py
    │   ├── reason.py
    │   └── search.py
    ├── firecrawl_client.py
    ├── main.py
    ├── orchestrator.py
    ├── output_manager.py
    └── perplexity_client.py
├── rat_agentic.py
└── setup.py
test_agent.py
```

### Files & Modules

1. **rat/__init__.py**
   - Package initialization
   - Exports the main function for convenience

2. **rat/research/__init__.py**
   - Exports main classes (PerplexityClient, FirecrawlClient, ResearchOrchestrator, OutputManager) for use outside the package

3. **rat/research/agents/**
   - `base.py`: Abstract BaseAgent class defining decision-making structures, concurrency handling, and logging
   - `context.py`: Structures to track research data (ResearchContext, ContextBranch, and ContentItem) including branching and merging
   - `explore.py`: Implementation of the ExploreAgent, which scrapes URLs and adds the extracted content to the context
   - `reason.py`: Implementation of the ReasoningAgent. Uses the Gemini model to analyze content, detect gaps, propose new searches, or decide to terminate
   - `search.py`: Implementation of the SearchAgent using Perplexity to perform web searches

4. **rat/research/firecrawl_client.py**
   - Handles the FirecrawlClient for scraping web pages
   - Wraps the Firecrawl API calls

5. **rat/research/perplexity_client.py**
   - Manages Perplexity queries for the SearchAgent

6. **rat/research/orchestrator.py**
   - Coordinates the flow of decisions
   - Executes them in multiple iterations
   - Uses the OutputManager to record intermediate and final results

7. **rat/research/output_manager.py**
   - Handles the creation of research directories
   - Saves context states, iteration metrics, and final outputs (Markdown & PDF)

8. **rat/research/main.py**
   - A command-line interface (CLI) entry point
   - Provides --interactive mode and direct usage

9. **rat_agentic.py**
   - Another CLI entry point with an interactive shell
   - Allows users to enter queries, see configurations, and run multi-agent research interactively

10. **test_agent.py**
    - A sample test script demonstrating how to instantiate the ResearchOrchestrator and run a single research question

11. **setup.py**
    - Python package setup, including a console_scripts entry point if you wish to install RAT as a system-wide command

## Workflow & Logic

### Step-by-Step Flow

1. **User Inputs a Question**
   - The main orchestrator (ResearchOrchestrator) is initialized with configuration
   - The user question is stored in the ResearchContext

2. **Iteration Loop**
   - ReasoningAgent proposes which actions to take (e.g., do a search, explore a specific URL, or finalize)
   - The orchestrator gathers decisions from SearchAgent and ExploreAgent as well
   - Decisions are sorted by priority:
     - SEARCH → Uses SearchAgent (Perplexity) to gather info
     - EXPLORE → Uses ExploreAgent (Firecrawl) to scrape content
     - REASON → Gemini-based analysis on newly acquired data
     - TERMINATE → End research

3. **Add New Content to Context**
   - After each decision is executed, the resulting data (search text, explored webpage content, analysis outcome) is added to the context (ContentItems)

4. **Check for Completion**
   - The loop continues until the ReasoningAgent decides there's enough info or no more actions are needed
   - The orchestrator can also terminate if no new decisions are generated (i.e., no further lines of inquiry)

5. **Final Output**
   - A consolidated research paper is generated in Markdown (and optionally exported to PDF)
   - The orchestrator and OutputManager store logs, context snapshots, and final metrics in a timestamped directory (research_outputs/<timestamp>_question/)

## Usage Instructions

### 1. Command-Line (Interactive)
```bash
python -m rat.research.main --interactive
```
Follow the prompts to type in a question or commands (config, metrics, help, exit).

### 2. Command-Line (Single Pass)
```bash
python -m rat.research.main "What are the health benefits of green tea?"
```
The system runs once, then outputs the final paper.

### 3. Alternative CLI (rat_agentic.py)
```bash
python rat_agentic.py
```
Enter your research query directly, or type quit to exit.

### 4. Installed Script
If installed via pip install ., you can run (assuming entry points are configured in setup.py):
```bash
rat-research "Your question here"
```

### 5. Programmatically
```python
from rat.research.orchestrator import ResearchOrchestrator

orchestrator = ResearchOrchestrator()
results = orchestrator.start_research("What are the main features of Billit's accounting software?")
print(results["paper"])  # or handle further
```

## Configuration

### .env File
Provide API keys:
```env
PERPLEXITY_API_KEY="..."
FIRECRAWL_API_KEY="..."
GEMINI_API_KEY="..."
```

### Environment Variables
Fine-tune limits like rate limits and max workers:
```bash
export SEARCH_RATE_LIMIT=100
export EXPLORE_RATE_LIMIT=50
export REASON_RATE_LIMIT=10
```

### Orchestrator Configuration
- `max_iterations`, `min_new_content`, `min_confidence`, etc. can be set in environment variables or overridden in code

## Logging & Outputs

- `rat.log`: General runtime logs (info level)
- `rat_api.log`: Detailed logs from external API calls (debug level)
- `research_outputs/`: Timestamped folder containing:
  - `research_paper.md` & `research_paper.pdf`
  - `metadata.json` & `research_info.json`
  - `iteration_metrics.json` & `states/` subfolder with context states

## Testing & Examples

### 1. test_agent.py
- Demonstrates how to run a quick test question and print results

### 2. Example Command
```bash
python test_agent.py
```
Runs the orchestrator on a sample question about Billit's accounting software.

### 3. Check Logs & Output
- After the script finishes, consult `rat.log`, `rat_api.log`, or the `research_outputs/` folder for detailed artifacts

## Contributing

- Pull Requests welcome!
- Issues: Report bugs or enhancements on the GitHub issue tracker
- Coding Conventions: Follows PEP 8 style guidelines

## License

MIT License - You are free to use, modify, and distribute this software with attribution.

---

Enjoy exploring with RAT! If you have any questions or run into issues, feel free to open an issue or contribute.