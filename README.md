# 🧠 RAT (Retrieval Augmented Thinking)

> *Enhancing AI responses through structured reasoning and knowledge retrieval*

RAT is a powerful tool that improves AI responses by leveraging DeepSeek's reasoning capabilities to guide other models through a structured thinking process.

## 💡 Origin & Ideation

The idea for RAT emerged from an interesting discovery about DeepSeek-R1 API capabilities. By setting the final response token to 1 while retrieving the thinking process, it became possible to separate the reasoning stage from the final response generation. This insight led to the development of a two-stage approach that combines DeepSeek's exceptional reasoning abilities with various response models.

Link to my original concept in this [Twitter thread](https://x.com/skirano/status/1881922469411643413).

## How It Works

RAT employs a two-stage approach:
1. **Reasoning Stage** (DeepSeek): Generates detailed reasoning and analysis for each query
2. **Response Stage** (OpenRouter): Utilizes the reasoning context to provide informed, well-structured answers

This approach ensures more thoughtful, contextually aware, and reliable responses.

## 🎯 Features

- 🤖 **Model Selection**: Flexibility to choose from various OpenRouter models
- 🧠 **Reasoning Visibility**: Toggle visibility of the AI's thinking
- 🔄 **Context Awareness**: Maintains conversation context for more coherent interactions

## ⚙️ Requirements

• Python 3.11 or higher  
• A .env file containing:
  ```plaintext
  DEEPSEEK_API_KEY=your_deepseek_api_key
  OPENROUTER_API_KEY=your_openrouter_api_key
  optional
  ANTHROPIC_API_KEY=your_anthropic_api_key_here
  ```

## 🚀 Installation
Standalone installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Doriandarko/RAT-retrieval-augmented-thinking.git
   cd RAT-retrieval-augmented-thinking
   ```


2. Install as a local package:
   ```bash
   pip install -e .
   ```

This will install RAT as a command-line tool, allowing you to run it from anywhere by simply typing `rat`!

## 📖 Usage

1. Ensure your .env file is configured with:
   ```plaintext
   DEEPSEEK_API_KEY=your_deepseek_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   optional
   ANTHROPIC_API_KEY=your_anthropic_api_key_here 
   ```

2. Run RAT from anywhere:
   ```bash
   rat
   ```

3. Available commands:
   - Enter your question to get a reasoned response
   - Use "model <name>" to switch OpenRouter models
   - Type "reasoning" to show/hide the thinking process
   - Type "quit" to exit



## 🚀 Versions
You can also run each script on its own:

### Standard Version (rat.py)
The default implementation using DeepSeek for reasoning and OpenRouter for responses.
Run it using:
```bash
uv run rat.py
```

### Claude-Specific Version (rat-claude.py)
A specialized implementation designed for Claude models that leverages Anthropic's message prefilling capabilities. This version makes Claude believe the reasoning process is its own internal thought process, leading to more coherent and contextually aware responses.
Run it using:
```bash
uv run rat-claude.py
```


## 🤝 Contributing

Interested in improving RAT?

1. Fork the repository
2. Create your feature branch
3. Make your improvements
4. Submit a Pull Request

## 📜 License

This project is available under the MIT License. See the [LICENSE](LICENSE) file for details.

If you use this codebase in your projects, please include appropriate credits:

```plaintext
This project uses RAT (Retrieval Augmented Thinking) by Skirano
GitHub: https://github.com/yourusername/rat
```
---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Doriandarko/RAT-retrieval-augmented-thinking&type=Date)](https://star-history.com/#Doriandarko/RAT-retrieval-augmented-thinking&Date)

# RAT Research - Enhanced Retrieval Augmented Thinking

RAT Research is an enhanced version of the Retrieval Augmented Thinking (RAT) system that combines DeepSeek's reasoning capabilities with web research through Perplexity API and Jina Reader. It generates comprehensive research papers in response to your questions.

## Features

- **DeepSeek Reasoning**: Analyzes research questions and plans the research approach
- **Perplexity Web Search**: Conducts intelligent web searches to gather information
- **Jina Reader Integration**: Extracts and processes content from web sources
- **Dynamic Paper Generation**: Creates well-structured research papers in markdown format
- **Multiple Output Formats**: Supports both corporate and academic paper styles
- **Citation Management**: Tracks and formats source citations
- **Interactive CLI**: User-friendly command-line interface with rich formatting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RAT-research.git
cd RAT-research
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up environment variables in `.env`:
```
DEEPSEEK_API_KEY=your_deepseek_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
JINA_API_KEY=your_jina_api_key
```

## Usage

Start the RAT Research CLI:
```bash
python -m rat.rat_research
```

### Commands

- Type your research question to start the research process
- `reasoning` - Toggle visibility of the reasoning process
- `export` - Export the last generated paper to a file
- `clear` - Clear research history
- `quit` - Exit the application

### Example

```
Research Question: What are the latest developments in quantum computing?

[Research Approach Reasoning]
1. First, we'll search for recent academic and industry sources...
2. Then, we'll analyze quantum computing breakthroughs...
3. Finally, we'll synthesize the findings into a comprehensive report...

[Conducting Research...]
...

[Generated Paper]
# Latest Developments in Quantum Computing

## Executive Summary
...
```

## Paper Formats

### Corporate Format
- Executive Summary
- Key Findings
- Detailed Analysis
- Recommendations
- Sources

### Academic Format
- Abstract
- Introduction
- Methodology
- Results
- Discussion
- Conclusion
- References

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built upon the original RAT system
- Uses DeepSeek for reasoning capabilities
- Integrates Perplexity API for web search
- Uses Jina Reader for content extraction

