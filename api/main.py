import os, json
from fastapi import FastAPI, HTTPException, Depends
from infrastructure.api.routes import stream, forecast, agent, grid, optimizer
from infrastructure.api.dependencies import get_predictor_dep, get_streaming_engine_dep

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="GridTokenX Clean API", version="3.0.0")

# Include routers
app.include_router(stream.router)
app.include_router(forecast.router)
app.include_router(agent.router)
app.include_router(grid.router)
app.include_router(optimizer.router)

@app.get("/health")
def health(
    predictor=Depends(get_predictor_dep),
    stream=Depends(get_streaming_engine_dep)
):
    return {
        "status": "ok", 
        "device": predictor.device, 
        "buffer": len(stream.buffer), 
        "window": stream.window_size
    }

@app.get("/metrics")
def metrics():
    path = os.path.join(ROOT, "results/evaluation_report.json")
    if not os.path.exists(path): 
        raise HTTPException(404, "Run research/evaluate.py first.")
    with open(path) as f: 
        return json.load(f)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
