import os
import mlflow
import shutil
from mlflow.tracking import MlflowClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

def deploy_production_models():
    client = MlflowClient()
    
    # Models to deploy
    models = {
        "GridTokenX_LGBM": "lgbm.pkl",
        "GridTokenX_TCN": "tcn.pt",
        "GridTokenX_MetaLearner": "meta_learner.pkl"
    }
    
    print("🚀 Deploying Production Models from MLflow Registry...")
    
    for model_name, target_file in models.items():
        try:
            # Get the latest version with the "Production" alias or just the latest version
            # Note: In MLflow 2.x, 'aliases' are used instead of 'stages'
            try:
                version_info = client.get_model_version_by_alias(model_name, "Production")
                print(f"  Fetching {model_name} (Alias: Production, Version: {version_info.version})...")
            except:
                # Fallback to latest version if no Production alias exists
                versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
                version_info = versions[0]
                print(f"  [Fallback] Fetching {model_name} (Latest, Version: {version_info.version})...")
            
            # Download artifact
            artifact_uri = client.get_model_version_download_uri(model_name, version_info.version)
            local_path = mlflow.artifacts.download_artifacts(artifact_uri)
            
            # The artifact_path was set during log_model (e.g., "lgbm_model")
            # We need to find the actual file inside the downloaded directory
            found_file = None
            for root, dirs, files in os.walk(local_path):
                # For sklearn/pickle models, they are often named model.pkl inside the artifact dir
                if "model.pkl" in files:
                    found_file = os.path.join(root, "model.pkl")
                    break
                # For pytorch, it might be named model.pth or similar
                if "model.pth" in files:
                    found_file = os.path.join(root, "model.pth")
                    break
                # Or custom filenames we saved
                if target_file in files:
                    found_file = os.path.join(root, target_file)
                    break

            if found_file:
                target_path = os.path.join(MODELS_DIR, target_file)
                shutil.copy2(found_file, target_path)
                print(f"  ✅ Deployed {target_file}")
            else:
                print(f"  ❌ Could not find model file in artifacts for {model_name}")
                
        except Exception as e:
            print(f"  ❌ Failed to deploy {model_name}: {e}")

    print("\n✅ Deployment cycle complete.")

if __name__ == "__main__":
    deploy_production_models()
