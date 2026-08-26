# 🤖 Agentic MLOps Pipeline

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agentic-mlops.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)

A **self-correcting multi-agent system** for MLOps using **LangGraph**. This project demonstrates autonomous ML pipelines with feedback loops where agents can reject and retry when quality standards aren't met.

![Pipeline Architecture](https://mermaid.ink/img/pako:eNp1kU1qwzAQha9izLoJzqJbQ6GrQqGrdCEkjSITW5LRyKSE3L2yHTdplsY8vj_maSR4qqMaGJXuqL3xyAG_OiJfewynBsMJkRrMVg8E3-Nig0K3hGZNaLdMaA4YTgjNhtDumNAeMJwRmh2h3TOhPWI4JzR7QntgQnvCcEFoDkzoCKE7Yjgj0IOhuyLSHBntOaE-EekBo70i0jtGe02kd4z2hkjvGe0tkR4w2jsiPWS0D0T6wGgfifSI0T4R6TGjfSbSE0b7QqSnjPaVSM8Y7RuRnjPadyK9YLQfRHrJaD-J9IrRfhHpNaP9JtJbRvtDpHeM9pdI7xjtPyJ9YLT/flowchart-TD-START-DE-MATH-CHECK-MLOPS-END?type=png)

## ✨ Features

- **Multi-Agent System**: 3 specialized agents (Data Engineer, Mathematician, MLOps)
- **Self-Correcting Loop**: Automatic retry with different strategies when validation fails
- **GridSearchCV**: Automated hyperparameter tuning
- **Production Ready**: Generates Dockerfile + FastAPI serving script
- **Interactive Dashboard**: Streamlit UI to visualize and run the pipeline

## 🏗️ Architecture

```
┌─────────────────────┐
│    DataEngineer     │ ◄──────────────────────┐
│   (clean data)      │                        │
└──────────┬──────────┘                        │
           │                                   │
           ▼                                   │
┌─────────────────────┐                        │
│    Mathematician    │                        │
│  (GridSearch, eval) │                        │
└──────────┬──────────┘                        │
           │                                   │
           ▼                                   │
     ┌───────────┐      accuracy < 70%         │
     │  Check    │─────────────────────────────┘
     └─────┬─────┘
           │ accuracy >= 70%
           ▼
┌─────────────────────┐
│       MLOps         │
│ (Docker, FastAPI)   │
└──────────┬──────────┘
           │
           ▼
       [ END ]
```

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/carlosperezvega/agentic-mlops-pipeline.git
cd agentic-mlops-pipeline

# Install dependencies
pip install -r requirements.txt

# Run the pipeline (CLI)
python main.py

# Run the Streamlit dashboard
streamlit run app.py
```

### Try the Demo

Visit the live demo: **[agentic-mlops.streamlit.app](https://agentic-mlops.streamlit.app)**

## 📁 Project Structure

```
agentic-mlops-pipeline/
├── app.py                  # Streamlit dashboard
├── main.py                 # CLI entry point
├── state.py                # AgentState (TypedDict)
├── agents/
│   ├── data_engineer.py    # Data cleaning agent
│   ├── mathematician.py    # Training & validation agent
│   └── mlops.py           # Containerization agent
├── tools/
│   ├── data_tools.py      # Pandas tools (@tool)
│   ├── ml_tools.py        # Sklearn tools (@tool)
│   └── devops_tools.py    # Docker tools (@tool)
├── data/
│   └── iris.csv           # Default dataset
├── models/                # Saved models (.pkl)
├── output/                # Generated Dockerfile & serve.py
└── .streamlit/
    └── config.toml        # Streamlit theme
```

## 🤖 Agents

| Agent | Responsibility | Tools Used |
|-------|---------------|------------|
| **Data Engineer** | Load CSV, clean nulls, prepare data | `load_csv_data`, `clean_null_values` |
| **Mathematician** | Train/test split, GridSearchCV, evaluate | `train_classifier`, `evaluate_model` |
| **MLOps Engineer** | Validate model, generate Dockerfile | `generate_dockerfile`, `save_model` |

## 📊 Technologies

- **LangGraph** - Multi-agent orchestration with conditional edges
- **LangChain** - Tool decorators and message handling
- **Scikit-learn** - RandomForest, GridSearchCV, metrics
- **Pandas** - Data processing
- **Streamlit** - Interactive dashboard
- **FastAPI** - Model serving API
- **Docker** - Containerization

## 🎯 Self-Correcting Loop

The pipeline implements automatic retry logic:

1. **Data Engineer** cleans data with strategy "mean"
2. **Mathematician** trains and evaluates model
3. If accuracy < 70%:
   - Feedback sent to Data Engineer
   - Retry with strategy "median"
   - If still failing, try "drop"
4. If accuracy >= 70%:
   - **MLOps** generates deployment artifacts

## 📈 Example Output

```
[DataEngineer] Loaded 150 rows, 5 columns
[DataEngineer] No null values found. Data is clean.
[Mathematician] Split: Train=120, Test=30 (stratified)
[Mathematician] GridSearch complete. Best params: {max_depth: 5, n_estimators: 100}
[Mathematician] Test metrics: Accuracy=0.9667, F1=0.9666
[MLOps] Model VALIDATED. Accuracy: 96.67% >= 70%
[MLOps] Dockerfile generated ✓
[MLOps] FastAPI serve.py generated ✓
```

## 🐳 Deploy Model

After running the pipeline:

```bash
cd output
docker build -t mlops-service .
docker run -p 8000:8000 mlops-service
```

API endpoints:
- `GET /health` - Health check
- `GET /info` - Model information
- `POST /predict` - Make predictions
- `GET /docs` - Swagger UI

## 👤 Author

**Carlos Perea Vega**
- Data Scientist & Mathematician
- Focus: MLOps, AI Agents, Google Cloud Platform

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
