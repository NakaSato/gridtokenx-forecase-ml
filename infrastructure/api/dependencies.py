import os
import yaml
from functools import lru_cache
from typing import Generator
from infrastructure.db.streaming_engine import SQLiteStreamingEngine
from infrastructure.ml.predictors import HybridPredictor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@lru_cache()
def get_config():
    config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

@lru_cache()
def get_predictor():
    return HybridPredictor(ROOT)

@lru_cache()
def get_streaming_engine():
    cfg = get_config()
    predictor = get_predictor()
    
    is_vercel = os.getenv("VERCEL") == "1"
    db_path = "/tmp/api_state.db" if is_vercel else os.path.join(ROOT, "api_state.db")
    
    # We need seq_fields from the predictor
    seq_fields = predictor.seq_fields
    window_size = predictor.window_size
    
    return SQLiteStreamingEngine(window_size, db_path, cfg, seq_fields)

# Dependency injection helpers for FastAPI
def get_predictor_dep():
    return get_predictor()

def get_streaming_engine_dep():
    return get_streaming_engine()

def get_config_dep():
    return get_config()
