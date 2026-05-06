import os
import mlflow

def setup_mlflow():
    """Configure MLflow tracking URI from environment or local sqlite."""
    # Use environment variable if set (e.g. in Docker or by user)
    # Fallback to local sqlite for ease of local development
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(os.getcwd(), 'mlflow.db')}")
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri
