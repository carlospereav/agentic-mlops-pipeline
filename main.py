"""
Agentic MLOps Pipeline - Main Entry Point

A self-correcting multi-agent system using LangGraph for MLOps workflows.
Implements the "Self-Correcting Loop" architecture with conditional edges
for backtracking when quality standards are not met.

Flow decisions (approve/reject) are made by conditional edge functions,
NOT by the agents themselves.
"""

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from agents.data_engineer import data_engineer_node
from agents.mathematician import mathematician_node
from agents.mlops import mlops_node
from state import AgentState


# Validation threshold for model approval
ACCURACY_THRESHOLD = 0.7

# Maximum retry attempts to prevent infinite loops
MAX_RETRIES = 3


def route_after_mathematician(state: AgentState) -> Literal["mlops", "data_engineer", "__end__"]:
    """
    Conditional edge: Decide next node after Mathematician.
    
    This function makes the flow decision based on metrics.
    The Mathematician agent itself does NOT make this decision.
    
    Args:
        state: Current AgentState with metrics.
        
    Returns:
        'mlops' if accuracy >= threshold,
        'data_engineer' if rejected and retries available,
        '__end__' if error or max retries exceeded.
    """
    status = state.get("status", "")
    metrics = state.get("metrics", {})
    messages = state.get("messages", [])
    
    # Handle error status
    if status == "error":
        return "__end__"
    
    # Check accuracy against threshold
    accuracy = metrics.get("accuracy", 0.0)
    
    if accuracy >= ACCURACY_THRESHOLD:
        # Model approved - proceed to MLOps
        return "mlops"
    else:
        # Model rejected - check retry count
        retry_count = _count_data_engineer_runs(messages)
        
        if retry_count < MAX_RETRIES:
            # Retry with different cleaning strategy
            return "data_engineer"
        else:
            # Max retries exceeded - end pipeline
            return "__end__"


def route_after_mlops(state: AgentState) -> Literal["mathematician", "__end__"]:
    """
    Conditional edge: Decide next node after MLOps.
    
    Args:
        state: Current AgentState.
        
    Returns:
        'mathematician' if dependency error requires retraining,
        '__end__' if successful or unrecoverable error.
    """
    status = state.get("status", "")
    
    if status == "rejected":
        # MLOps rejected due to low metrics - retry mathematician
        return "mathematician"
    else:
        # Success or error - end pipeline
        return "__end__"


def _count_data_engineer_runs(messages: list) -> int:
    """Count how many times Data Engineer has run based on messages."""
    count = 0
    for msg in messages:
        if hasattr(msg, "content") and "[DataEngineer] Starting" in msg.content:
            count += 1
        elif hasattr(msg, "content") and "[DataEngineer] Feedback received" in msg.content:
            count += 1
    return count


def create_mlops_graph() -> StateGraph:
    """
    Create the MLOps StateGraph with conditional edges for self-correction.
    
    Architecture:
        START -> DataEngineer -> Mathematician --(conditional)--> MLOps -> END
                      ^                |                           |
                      |   (rejected)   |                           |
                      +----------------+                           |
                      ^                    (dependency_error)      |
                      +--------------------------------------------+
    
    Returns:
        Compiled StateGraph ready for execution.
    """
    # Initialize the graph with our state schema
    workflow = StateGraph(AgentState)
    
    # Add nodes (agents)
    workflow.add_node("data_engineer", data_engineer_node)
    workflow.add_node("mathematician", mathematician_node)
    workflow.add_node("mlops", mlops_node)
    
    # Set entry point
    workflow.set_entry_point("data_engineer")
    
    # Add edges
    # Data Engineer -> Mathematician (always)
    workflow.add_edge("data_engineer", "mathematician")
    
    # Mathematician -> conditional (MLOps or Data Engineer or END)
    workflow.add_conditional_edges(
        "mathematician",
        route_after_mathematician,
        {
            "mlops": "mlops",
            "data_engineer": "data_engineer",
            "__end__": END,
        }
    )
    
    # MLOps -> conditional (Mathematician or END)
    workflow.add_conditional_edges(
        "mlops",
        route_after_mlops,
        {
            "mathematician": "mathematician",
            "__end__": END,
        }
    )
    
    return workflow.compile()


def run_pipeline(data_path: str) -> dict[str, Any]:
    """
    Execute the MLOps pipeline on a given dataset.
    
    Args:
        data_path: Path to the input CSV file.
        
    Returns:
        Final state dictionary with all results.
    """
    # Create the graph
    app = create_mlops_graph()
    
    # Initialize the state
    initial_state: AgentState = {
        "data_path": data_path,
        "model_path": "",
        "model": None,
        "metrics": {},
        "feedback": "",
        "messages": [],
        "status": "starting",
    }
    
    # Run the pipeline
    final_state = app.invoke(initial_state)
    
    return final_state


def print_graph_ascii() -> None:
    """Print ASCII visualization of the graph structure."""
    print("\n" + "=" * 70)
    print("AGENTIC MLOPS PIPELINE - SELF-CORRECTING LOOP")
    print("=" * 70)
    print("""
    ┌─────────────────────┐
    │    DataEngineer     │ ◄───────────────────────────────┐
    │  (clean, split)     │                                 │
    └──────────┬──────────┘                                 │
               │                                            │
               ▼                                            │
    ┌─────────────────────┐                                 │
    │    Mathematician    │                                 │
    │  (train, evaluate)  │                                 │
    └──────────┬──────────┘                                 │
               │                                            │
               ▼                                            │
    ┌─────────────────────┐     accuracy < 0.7              │
    │  Conditional Edge   │─────────────────────────────────┘
    │  (route decision)   │
    └──────────┬──────────┘
               │ accuracy >= 0.7
               ▼
    ┌─────────────────────┐
    │       MLOps         │
    │ (dockerfile, serve) │
    └──────────┬──────────┘
               │
               ▼
           [ END ]
    """)
    print("=" * 70)
    print(f"Accuracy Threshold: {ACCURACY_THRESHOLD:.0%}")
    print(f"Max Retries: {MAX_RETRIES}")
    print("=" * 70 + "\n")


def print_pipeline_results(state: dict[str, Any]) -> None:
    """
    Print formatted results from a pipeline run.
    
    Args:
        state: Final state dictionary from pipeline execution.
    """
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION RESULTS")
    print("=" * 70)
    
    # Print all messages
    print("\n📋 EXECUTION LOG:")
    print("-" * 50)
    for msg in state.get("messages", []):
        print(f"  {msg.content}")
    
    # Print final status
    print("\n📊 FINAL STATUS:")
    print("-" * 50)
    print(f"  Status: {state.get('status', 'unknown')}")
    print(f"  Model Path: {state.get('model_path', 'N/A')}")
    
    # Print metrics if available
    metrics = state.get("metrics", {})
    if metrics:
        print("\n📈 MODEL METRICS:")
        print("-" * 50)
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    # Print feedback if any
    feedback = state.get("feedback", "")
    if feedback:
        print("\n⚠️ FEEDBACK:")
        print("-" * 50)
        print(f"  {feedback}")
    
    print("\n" + "=" * 70)


# Entry point
if __name__ == "__main__":
    import sys
    
    # Print graph structure
    print_graph_ascii()
    
    # Check for data path argument
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        print(f"Running pipeline on: {data_file}")
        print("-" * 50)
        
        result = run_pipeline(data_file)
        print_pipeline_results(result)
    else:
        print("Usage: python main.py <path_to_csv>")
        print("\nExample:")
        print("  python main.py data/dataset.csv")
        print("\nNote: CSV file must be in allowed directories (data/, datasets/, input/)")
