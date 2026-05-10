from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import create_engine, text
from domain.entities import ClusterDispatchRequest
from infrastructure.api.dependencies import get_config_dep
from optimizer.cluster_dispatch_admm import get_cluster_dispatch
import os

router = APIRouter(tags=["grid"])

# Initialize DB engine lazily
POSTGIS_URL = os.getenv("POSTGIS_URL", "postgresql://gridtokenx:gridtokenx_pass@localhost:5432/gridtokenx_geo")
_pg_engine = None

def get_pg_engine():
    global _pg_engine
    if _pg_engine is None:
        try:
            _pg_engine = create_engine(POSTGIS_URL, echo=False)
        except Exception as e:
            print(f"⚠️  PostGIS connection failed: {e}")
            return None
    return _pg_engine

@router.get("/grid/assets")
def get_grid_assets(table: str = Query("egat_power_plants"), limit: int = 100):
    """Retrieve spatial assets from PostGIS as GeoJSON."""
    engine = get_pg_engine()
    if not engine:
        raise HTTPException(503, "PostGIS database not available")
    
    # Whitelist allowed tables for security
    allowed = ["egat_power_plants", "egat_substations", "power_plants", "egat_lines", "egat_towers", "koh_samui_grid"]
    if table not in allowed:
        raise HTTPException(400, f"Table '{table}' not accessible or does not exist")

    try:
        with engine.connect() as conn:
            query = text(f"""
                SELECT jsonb_build_object(
                    'type',     'FeatureCollection',
                    'features', COALESCE(jsonb_agg(features.feature), '[]'::jsonb)
                )
                FROM (
                  SELECT jsonb_build_object(
                    'type',       'Feature',
                    'id',         id,
                    'geometry',   ST_AsGeoJSON(geom)::jsonb,
                    'properties', to_jsonb(inputs) - 'geom' - 'id'
                  ) AS feature
                  FROM (SELECT * FROM {table} LIMIT :limit) inputs
                ) features;
            """)
            result = conn.execute(query, {"limit": limit}).scalar()
            return result or {"type": "FeatureCollection", "features": []}
    except Exception as e:
        raise HTTPException(500, f"Database error: {str(e)}")

@router.post("/dispatch/cluster")
def dispatch_cluster(req: ClusterDispatchRequest):
    """Run ADMM multi-island diesel coordination."""
    try:
        result = get_cluster_dispatch(
            samui_load=req.samui_load_mw,
            phangan_load=req.phangan_load_mw,
            tao_load=req.tao_load_mw
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"ADMM Dispatch failed: {str(e)}")
