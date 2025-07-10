
# 🧠 Project Blueprint for Claude 4

> Paste this into Claude in Cursor.com and say:  
> "**Use this blueprint to build the entire project in Python. I want everything built automatically by you.**"

---

## 🔧 Project Name:  
**LogNarrator AI**

---

## 📝 What This Tool Does:
This tool reads raw machine/tool logs (pasted or uploaded as .txt), analyzes sequences of events, categorizes them into operational phases, detects failures or retries, and **outputs a readable summary of what happened**, using contextual AI analysis.

---

## 🎯 Main Goals:
1. **Automatically parse logs** with timestamps, subsystems, and events.
2. **Categorize log events** into meaningful phases: INIT, POSITION, SCAN, SAVE, ERROR, RECOVERY, ABORT.
3. **Detect chains** like retries, failures, and slowdowns.
4. **Explain in plain English** what’s happening in the log.
5. **Highlight issues or risks** using a risk level (green/yellow/red).
6. **Show the full timeline** in a readable and structured way.

---

## 🔄 Input Format:
**Raw logs** (copy-pasted or from `.txt` file), structured like this:

```
00:39:24.243 Save Engine Async Save Triggered  
00:39:24.267 AF System Retry #1 Triggered  
00:39:26.214 SEM Image discarded
```

---

## ✅ Output Format:
1. A **table** showing:
   - Timestamp
   - Subsystem
   - Event
   - Categorized Phase
   - Risk Level (🟢 🟡 🔴)
   - Plain-English Explanation

2. A **narrative summary**:
   > "The tool attempted to save an image, but auto-focus failed and triggered a retry. Eventually, the image was discarded. This indicates a failed acquisition sequence and likely timing mismatch."

---

## 🧱 Architecture Breakdown:

### 🧩 1. Log Parser
Regex to split each log line:
```
(\d{2}:\d{2}:\d{2}\.\d{3})\s+(\w[\w\s]+?)\s+(.+)
```
Extract:
- `timestamp`
- `subsystem`
- `event description`

---

### 🧠 2. AI Categorizer (Phase Detection)
Map events to phases:
```python
PHASES = {
  "Init": ["Triggered", "Start"],
  "Position": ["Position", "Align"],
  "Scan": ["Scan", "Grab"],
  "Save": ["Save", "Stored"],
  "Error": ["discarded", "error", "failed"],
  "Recovery": ["Retry", "Recovered"],
  "Abort": ["Abort", "Halted"]
}
```

---

### 🚨 3. Risk Classifier
Assign risk levels:
- 🔴 Red: `discarded`, `failed`, `abort`
- 🟡 Yellow: `retry`, `stall`, `recovered`
- 🟢 Green: `save`, `scan`, `normal`

---

### 🗣️ 4. Natural Language Generator
Use Claude 4 to generate explanations from parsed logs like:
```python
"AF System retried due to failure. Save Engine triggered async save. SEM discarded image. This suggests acquisition failure due to timing mismatch."
```

---

### 🧾 5. Output Renderer (Command Line or Web)
**Basic CLI output** or optional:
- Streamlit interface (upload file → view timeline → get summary)
- Export to `.txt` or `.json`

---

## 📂 Project Structure

```
/log_narrator
│
├── main.py             # Main entry point
├── parser.py           # Log file parser
├── categorizer.py      # Phase and risk classifier
├── summarizer.py       # Claude-powered explainer
├── data/sample.log     # Sample log file
├── requirements.txt    # Libraries (openai, streamlit, etc.)
```

---

## 📦 Python Packages to Use

```txt
openai         # Or anthropic for Claude API
streamlit      # (optional GUI)
pandas         # Table rendering
```

---

## 🧪 Sample Test Log for Development:

```
00:01:10.123 Save Engine Async Save Triggered  
00:01:10.245 AF System Retry #1 Triggered  
00:01:12.400 SEM Image discarded  
00:01:13.001 AF System Retry #2 Triggered  
00:01:14.520 AF System Recovery Complete  
```

---

## 💡 Claude’s Main Tasks:
- Write **parser code** for `.log` or pasted input.
- Implement **phase categorization logic**.
- Add **risk classification** and **icon display**.
- Generate **summaries** using AI prompt templates.
- (Optional) Build **streamlit UI** with upload box + table + summary pane.

---

## 🧠 Prompt Template to Feed Claude Internally:
> “Given this log timeline:  
> Timestamp: 00:01:10.123 | Subsystem: Save Engine | Event: Async Save Triggered  
> Timestamp: 00:01:10.245 | Subsystem: AF System | Event: Retry #1 Triggered  
> …  
> Generate a short, plain-English explanation of what this tool is doing and any issues or anomalies you detect.”

---

## 🔚 Final Output Should:
- Allow a user to paste in logs
- Show a human-readable timeline with phases + explanations
- Help people who don’t understand log files make fast decisions

---

## 🧰 Tech Stack Overview

### 🐍 Backend Language
- **Python** (core logic, parsing, classification, reasoning)

---

### 📦 Key Python Libraries

| Library      | Purpose |
|--------------|---------|
| `re`         | For parsing log lines with regular expressions |
| `datetime`   | Handling and comparing timestamps |
| `pandas`     | Tabular data organization and export (optional, but useful) |
| `openai` or `anthropic` | API access for GPT or Claude summaries |
| `streamlit`  | (Optional) Build a lightweight browser UI for file upload & display |
| `json`       | (Optional) For exporting structured summaries or logs |
| `typer` or `argparse` | (Optional) CLI interface for power users |

---

### 🧠 AI Agent / Model
- **Claude 4** via **Anthropic API** (preferred)
  - Input: structured logs or sequences
  - Output: summaries, risk explanations, sequence interpretation

> Alternatively: GPT-4 via OpenAI API for fallback or parallel use

---

### 💻 Development Environment
- **Cursor.com** (AI-powered IDE with Claude 4 as agent/coder)
- Supports Markdown docs, Python file editing, AI commands

---

### 🖥️ Frontend (Optional)
If you choose to add a GUI for internal team use:
- **Streamlit**
  - Upload `.txt` files
  - View parsed timelines and risk table
  - See AI-generated summaries

---

### 📂 File Structure Example

```
/log_narrator
├── main.py               # Entry point
├── parser.py             # Log parsing logic
├── categorizer.py        # Phase and risk logic
├── summarizer.py         # Claude/GPT API prompts
├── ui_streamlit.py       # Optional frontend
├── requirements.txt      # Dependencies
├── data/sample.log       # Sample input
```

---

## ✅ Stack Summary

| Layer        | Tech                             |
|--------------|----------------------------------|
| Language     | Python                           |
| IDE          | Cursor.com + Claude 4            |
| Backend AI   | Claude 4 API                     |
| Frontend UI  | Streamlit (optional)             |
| Parser       | Regex + datetime                 |
| Display      | Terminal or browser table (via Streamlit) |
