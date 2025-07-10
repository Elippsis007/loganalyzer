# 🤖 LogNarrator AI

Transform raw machine/tool logs into human-readable summaries with AI-powered analysis.

## 🎯 What It Does

LogNarrator AI automatically analyzes raw log files and provides:
- **Parsed timeline** of events with timestamps, subsystems, and descriptions
- **Smart categorization** into operational phases (Init, Position, Scan, Save, Error, Recovery, Abort)
- **Risk assessment** with color-coded indicators (🟢 🟡 🔴)
- **AI-powered summaries** in plain English
- **Actionable recommendations** for identified issues
- **Interactive visualizations** (web interface only)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Basic Usage (CLI)

```bash
# Analyze a log file
python main.py -f data/sample.log

# Interactive mode (paste logs)
python main.py -i

# Show detailed timeline
python main.py -f data/sample.log -d

# Save summary to file
python main.py -f data/sample.log -o analysis.txt

# Pipe log data
cat logfile.txt | python main.py
```

### 3. Web Interface

```bash
# Launch Streamlit web interface
streamlit run ui_streamlit.py
```

Then open `http://localhost:8501` in your browser.

## 🔧 Configuration

### API Key (Optional but Recommended)

For enhanced AI analysis, configure your Anthropic API key:

```bash
# Set environment variable
export ANTHROPIC_API_KEY="your_api_key_here"

# Or use command line flag
python main.py -f logfile.txt --api-key "your_api_key_here"
```

Without an API key, the system uses rule-based analysis which still provides good results.

## 📊 Features

### 🔍 Log Parsing
- Extracts timestamps, subsystems, and events using regex patterns
- Handles multiple log formats automatically
- Provides parsing statistics and error reporting

### 📈 Categorization
- **Phases**: Init, Position, Scan, Save, Error, Recovery, Abort
- **Risk Levels**: 🟢 Green (normal), 🟡 Yellow (warning), 🔴 Red (critical)
- **Context-aware** rules for improved accuracy

### 🤖 AI Analysis
- Plain-English timeline summaries
- Risk assessment with percentages
- Actionable recommendations based on detected patterns
- Technical details and statistics

### 💻 Interfaces
- **CLI**: Command-line interface for scripting and automation
- **Web**: Interactive Streamlit interface with visualizations
- **Rich output**: Colored terminal output with tables and panels

## 📝 Log Format

The system works with logs in this format:

```
HH:MM:SS.mmm SubsystemName Event Description
```

Examples:
```
00:39:24.243 Save Engine Async Save Triggered
00:39:24.267 AF System Retry #1 Triggered
00:39:26.214 SEM Image discarded
```

## 🧪 Try It Out

Test with the included sample data:

```bash
# CLI analysis
python main.py -f data/sample.log

# Web interface with sample data
streamlit run ui_streamlit.py
# Click "Try with Sample Data" button
```

## 📚 CLI Options

```
usage: main.py [-h] [-f FILE] [-i] [-d] [-o OUTPUT] [--api-key API_KEY]

LogNarrator AI - Transform raw logs into readable summaries

options:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Log file to analyze
  -i, --interactive     Interactive mode
  -d, --detailed        Show detailed timeline
  -o OUTPUT, --output OUTPUT
                        Output file for summary
  --api-key API_KEY     Anthropic API key for enhanced AI analysis
```

## 🏗️ Architecture

```
logAnalyzer/
├── main.py              # CLI interface
├── parser.py            # Log parsing logic
├── categorizer.py       # Phase & risk classification
├── summarizer.py        # AI-powered summaries
├── ui_streamlit.py      # Web interface
├── requirements.txt     # Dependencies
├── data/
│   └── sample.log      # Sample log file
└── README.md           # This file
```

## 🔧 Components

### Parser (`parser.py`)
- Regex-based log line parsing
- Multiple format support
- Error handling and statistics

### Categorizer (`categorizer.py`)
- Phase detection using keyword matching
- Risk level classification
- Subsystem-specific rules
- Context-aware improvements

### Summarizer (`summarizer.py`)
- AI-powered narrative generation
- Risk assessment analysis
- Recommendation engine
- Technical details extraction

### CLI Interface (`main.py`)
- Command-line argument parsing
- Rich-formatted output
- File I/O and piping support
- Interactive mode

### Web Interface (`ui_streamlit.py`)
- File upload and text input
- Interactive visualizations
- Real-time analysis
- Export functionality

## 🎨 Example Output

```
=== LOG ANALYSIS SUMMARY ===

Overall Status: 🔄 Recovered

Timeline Summary:
The system initiated a save operation but encountered 1 error(s) with 2 warning(s). 
The system attempted recovery and completed successfully.

Key Events:
  🔴 00:39:26.214: SEM Image discarded
  🟡 00:39:24.267: AF System Retry #1 Triggered
  🟡 00:39:27.001: AF System Retry #2 Triggered

Risk Assessment:
MEDIUM RISK: 16.7% critical errors detected. Monitor system closely.

Recommendations:
  • Investigate and resolve critical errors before continuing operation
  • Check system stability - multiple retries indicate potential hardware issues
  • Check imaging parameters and sample conditions

Technical Details:
Duration: 00:39:24.243 to 00:39:58.000
Subsystems involved: AF System, Control System, Positioning Stage, SEM, Save Engine
Operational phases: Error, Recovery, Save, Scan
Total events: 28
```

## 🛠️ Development

### Adding New Patterns

1. **Phase Detection**: Edit `categorizer.py` → `phase_keywords`
2. **Risk Classification**: Edit `categorizer.py` → `risk_keywords`
3. **Subsystem Rules**: Edit `categorizer.py` → `subsystem_rules`

### Testing

```bash
# Test individual components
python parser.py
python categorizer.py
python summarizer.py

# Test with sample data
python main.py -f data/sample.log -d
```

## 📋 Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`
- Optional: Anthropic API key for enhanced AI analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use and modify as needed.

## 🆘 Support

For issues or questions:
1. Check the sample data and examples
2. Review the documentation
3. Test with different log formats
4. Verify API key configuration (if using AI features)

---

**Happy Log Analyzing!** 🎉 