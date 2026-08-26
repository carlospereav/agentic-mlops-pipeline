"""
Streamlit Dashboard for Agentic MLOps Pipeline
Interactive visualization of the self-correcting multi-agent system.
"""

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Page config
st.set_page_config(
    page_title="Agentic MLOps Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap');
    
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    
    .sub-header {
        font-family: 'Space Grotesk', sans-serif;
        color: #6b7280;
        font-size: 1.1rem;
        margin-top: 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #334155;
    }
    
    .agent-box {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .agent-data { border-color: #3b82f6; }
    .agent-math { border-color: #8b5cf6; }
    .agent-mlops { border-color: #10b981; }
    
    .log-container {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        background: #0d1117;
        border-radius: 8px;
        padding: 1rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .log-data { color: #58a6ff; }
    .log-math { color: #a371f7; }
    .log-mlops { color: #3fb950; }
    .log-error { color: #f85149; }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render the main header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="main-header">🤖 Agentic MLOps Pipeline</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Self-correcting multi-agent system with LangGraph</p>', unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="text-align: right; padding-top: 1rem;">
            <a href="https://github.com/carlosperezvega" target="_blank" style="color: #667eea; text-decoration: none;">
                👤 Carlos Perea Vega
            </a>
        </div>
        """, unsafe_allow_html=True)


def render_architecture():
    """Render the pipeline architecture diagram."""
    st.markdown("### 📐 Architecture: Self-Correcting Loop")
    
    mermaid_code = """
    ```mermaid
    flowchart TD
        START([🚀 START]) --> DE[🔧 DataEngineer<br/>Clean & Prepare]
        DE --> MATH[🧮 Mathematician<br/>Train & Validate]
        MATH --> CHECK{📊 Accuracy >= 70%?}
        CHECK -->|❌ NO| DE
        CHECK -->|✅ YES| MLOPS[🐳 MLOps<br/>Containerize]
        MLOPS --> END([🎯 END])
        
        style DE fill:#1e40af,stroke:#3b82f6,color:#fff
        style MATH fill:#6d28d9,stroke:#8b5cf6,color:#fff
        style MLOPS fill:#047857,stroke:#10b981,color:#fff
        style CHECK fill:#92400e,stroke:#f59e0b,color:#fff
    ```
    """
    st.markdown(mermaid_code)
    
    # Fallback text diagram
    with st.expander("📝 Text Diagram (if Mermaid doesn't render)"):
        st.code("""
    ┌─────────────────────┐
    │    DataEngineer     │ ◄──────────────┐
    │   (clean, prep)     │                │
    └──────────┬──────────┘                │
               │                           │
               ▼                           │
    ┌─────────────────────┐                │
    │    Mathematician    │                │
    │  (GridSearch, eval) │                │
    └──────────┬──────────┘                │
               │                           │
               ▼                           │
         ┌───────────┐    accuracy < 70%   │
         │  Check    │─────────────────────┘
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
        """, language=None)


def render_agents_info():
    """Render information about each agent."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="agent-box agent-data">
            <h4>🔧 Data Engineer</h4>
            <ul>
                <li>Loads CSV data</li>
                <li>Cleans null values</li>
                <li>Tries different strategies on feedback</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="agent-box agent-math">
            <h4>🧮 Mathematician</h4>
            <ul>
                <li>Train/test split</li>
                <li>GridSearchCV optimization</li>
                <li>Calculates metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="agent-box agent-mlops">
            <h4>🐳 MLOps Engineer</h4>
            <ul>
                <li>Validates model</li>
                <li>Generates Dockerfile</li>
                <li>Creates FastAPI serve.py</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to data directory."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    file_path = data_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)


def run_pipeline_with_progress(data_path: str):
    """Run the pipeline with progress updates."""
    from main import create_mlops_graph
    from state import AgentState
    
    # Initialize state
    initial_state: AgentState = {
        "data_path": data_path,
        "model_path": "",
        "model": None,
        "metrics": {},
        "feedback": "",
        "messages": [],
        "status": "starting",
    }
    
    # Create and run graph
    app = create_mlops_graph()
    final_state = app.invoke(initial_state)
    
    # Print logs to console for debugging
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION LOG (Console)")
    print("=" * 60)
    for msg in final_state.get("messages", []):
        if hasattr(msg, "content"):
            print(f"  {msg.content}")
    
    # Print metrics summary
    metrics = final_state.get("metrics", {})
    if metrics:
        print("-" * 60)
        print("METRICS:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    
    print(f"STATUS: {final_state.get('status', 'unknown')}")
    print("=" * 60 + "\n")
    
    return final_state


def format_log_message(content: str) -> str:
    """Format log message with color coding."""
    if "[DataEngineer]" in content:
        return f'<span class="log-data">{content}</span>'
    elif "[Mathematician]" in content:
        return f'<span class="log-math">{content}</span>'
    elif "[MLOps]" in content:
        return f'<span class="log-mlops">{content}</span>'
    elif "Error" in content or "error" in content:
        return f'<span class="log-error">{content}</span>'
    return content


def render_results(state: dict):
    """Render pipeline results."""
    st.markdown("---")
    st.markdown("### 📊 Pipeline Results")
    
    # Status indicator
    status = state.get("status", "unknown")
    status_colors = {
        "packaged": "🟢",
        "trained": "🟡",
        "cleaned": "🔵",
        "error": "🔴",
        "rejected": "🟠",
    }
    st.markdown(f"**Status:** {status_colors.get(status, '⚪')} `{status}`")
    
    # Metrics
    metrics = state.get("metrics", {})
    if metrics:
        st.markdown("#### 📈 Model Metrics")
        cols = st.columns(5)
        
        metric_icons = {
            "accuracy": "🎯",
            "precision": "📍",
            "recall": "🔄",
            "f1_score": "⚖️",
            "mae": "📏",
        }
        
        for i, (metric, value) in enumerate(metrics.items()):
            with cols[i % 5]:
                icon = metric_icons.get(metric, "📊")
                if metric == "mae":
                    st.metric(f"{icon} {metric.upper()}", f"{value:.4f}")
                else:
                    st.metric(f"{icon} {metric.upper()}", f"{value:.2%}")
    
    # Execution log
    st.markdown("#### 📋 Execution Log")
    messages = state.get("messages", [])
    
    log_html = '<div class="log-container">'
    for msg in messages:
        if hasattr(msg, "content"):
            formatted = format_log_message(msg.content)
            log_html += f"<div style='margin: 4px 0;'>{formatted}</div>"
    log_html += "</div>"
    
    st.markdown(log_html, unsafe_allow_html=True)
    
    # Model path
    model_path = state.get("model_path", "")
    if model_path:
        st.markdown(f"**Model saved to:** `{model_path}`")
    
    return metrics


def render_prediction_section(model_path: str):
    """Render the prediction section."""
    st.markdown("---")
    st.markdown("### 🔮 Make Predictions")
    
    if not model_path or not Path(model_path).exists():
        st.warning("No model available. Run the pipeline first.")
        return
    
    import joblib
    model = joblib.load(model_path)
    
    # Get feature names if available
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    if hasattr(model, "feature_names_in_"):
        feature_names = model.feature_names_in_.tolist()
    
    st.markdown("Enter feature values:")
    
    cols = st.columns(len(feature_names))
    input_values = {}
    
    for i, feature in enumerate(feature_names):
        with cols[i]:
            input_values[feature] = st.number_input(
                feature, 
                value=5.0, 
                step=0.1,
                format="%.2f",
                key=f"pred_{feature}"
            )
    
    if st.button("🔮 Predict", type="primary"):
        input_df = pd.DataFrame([input_values])
        prediction = model.predict(input_df)[0]
        
        # Get probabilities if available
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Prediction:** Class `{prediction}`")
        
        if proba is not None:
            with col2:
                st.markdown("**Probabilities:**")
                for i, p in enumerate(proba):
                    st.progress(p, text=f"Class {i}: {p:.2%}")


def render_files_generated():
    """Show generated files."""
    st.markdown("---")
    st.markdown("### 📁 Generated Files")
    
    col1, col2 = st.columns(2)
    
    # Dockerfile
    dockerfile_path = Path("output/Dockerfile")
    if dockerfile_path.exists():
        with col1:
            with st.expander("🐳 Dockerfile"):
                st.code(dockerfile_path.read_text(), language="dockerfile")
    
    # Serve.py
    serve_path = Path("output/serve.py")
    if serve_path.exists():
        with col2:
            with st.expander("🚀 serve.py (FastAPI)"):
                st.code(serve_path.read_text()[:3000] + "\n...", language="python")
    
    # Model metadata
    metadata_path = Path("models/trained_model_metadata.json")
    if metadata_path.exists():
        with st.expander("📊 Model Metadata"):
            metadata = json.loads(metadata_path.read_text())
            st.json(metadata)


def main():
    """Main Streamlit app."""
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Dataset selection
        st.markdown("### 📁 Dataset")
        data_option = st.radio(
            "Choose data source:",
            ["Use Iris (default)", "Upload CSV"],
            index=0
        )
        
        data_path = "data/iris.csv"
        
        if data_option == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload your CSV",
                type=["csv"],
                help="CSV with features and a 'target' column"
            )
            if uploaded_file:
                data_path = save_uploaded_file(uploaded_file)
                st.success(f"Uploaded: {uploaded_file.name}")
        
        # Show dataset preview
        if Path(data_path).exists():
            with st.expander("👀 Preview Data"):
                df = pd.read_csv(data_path)
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"{len(df)} rows × {len(df.columns)} columns")
        
        st.markdown("---")
        
        # Pipeline settings
        st.markdown("### 🎚️ Settings")
        st.info("Accuracy threshold: **70%**\nMax retries: **3**")
        
        st.markdown("---")
        st.markdown("""
        ### 🔗 Links
        - [GitHub](https://github.com/carlosperezvega)
        - [LinkedIn](https://linkedin.com/in/carlosperezvega)
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🏗️ Architecture", "▶️ Run Pipeline", "🔮 Predictions"])
    
    with tab1:
        render_architecture()
        st.markdown("---")
        render_agents_info()
    
    with tab2:
        st.markdown("### ▶️ Execute Pipeline")
        st.markdown(f"**Dataset:** `{data_path}`")
        
        if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
            with st.spinner("Running pipeline..."):
                progress_bar = st.progress(0, text="Initializing...")
                
                try:
                    progress_bar.progress(10, text="Loading data...")
                    time.sleep(0.3)
                    
                    progress_bar.progress(30, text="Data Engineer processing...")
                    time.sleep(0.3)
                    
                    progress_bar.progress(50, text="Mathematician training (GridSearchCV)...")
                    
                    # Run actual pipeline
                    result = run_pipeline_with_progress(data_path)
                    
                    progress_bar.progress(80, text="MLOps packaging...")
                    time.sleep(0.3)
                    
                    progress_bar.progress(100, text="Complete!")
                    time.sleep(0.2)
                    progress_bar.empty()
                    
                    # Store result in session state
                    st.session_state["pipeline_result"] = result
                    st.session_state["model_path"] = result.get("model_path", "")
                    
                    st.success("✅ Pipeline completed successfully!")
                    
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"❌ Pipeline failed: {str(e)}")
        
        # Show results if available
        if "pipeline_result" in st.session_state:
            render_results(st.session_state["pipeline_result"])
            render_files_generated()
    
    with tab3:
        model_path = st.session_state.get("model_path", "models/trained_model.pkl")
        render_prediction_section(model_path)


if __name__ == "__main__":
    main()

