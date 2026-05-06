"""
GridTokenX — Load GeoJSON spatial data into PostGIS
====================================================
Batch-loads all GeoJSON files from the geojson/ directory into PostGIS
GEOGRAPHY tables for spatial querying.

Uses cleaned/ variants when available (normalized schemas), falls back
to raw files otherwise.

Usage:
    uv run --extra geo python data/load_postgis.py [--url <POSTGIS_URL>]
    uv run --extra geo python data/load_postgis.py --tables power_plants,egat_substations
    just postgis-load
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
from geoalchemy2 import Geography, Geometry
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    inspect,
    text,
)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_URL = "postgresql://gridtokenx:gridtokenx_pass@localhost:5432/gridtokenx_geo"
GEOJSON_DIR = Path(__file__).parent / "geojson"
CLEANED_DIR = GEOJSON_DIR / "cleaned"
POWER_PLANTS_DIR = GEOJSON_DIR / "power_plants"

# ── Table Registry ────────────────────────────────────────────────────────────
# Each entry: table_name -> {geojson path, column_map, geometry type}
# Prefer cleaned/ files for normalized schemas where available.
# Files > 100 MB are marked optional (skipped unless explicitly requested).

GEOJSON_TYPE_MAP = {
    "Point": "POINT",
    "MultiPoint": "MULTIPOINT",
    "LineString": "LINESTRING",
    "MultiLineString": "MULTILINESTRING",
    "Polygon": "POLYGON",
    "MultiPolygon": "MULTIPOLYGON",
}

# ── Table definitions ─────────────────────────────────────────────────────────
# Organized by priority tier:
#   tier 1 = core grid assets (always loaded)
#   tier 2 = supplementary (always loaded, slightly larger)
#   tier 3 = heavy datasets (loaded only when explicitly requested via --all)

TABLE_REGISTRY = {
    # ── Tier 1: Core grid assets (small, essential) ──────────────────────────
    "power_plants": {
        "path": POWER_PLANTS_DIR / "thailand_generators.geojson",
        "column_map": {
            "Plant / Project name": "plant_name",
            "Type": "type",
            "Capacity (MW)": "capacity_mw",
            "Status": "status",
            "Operator": "operator",
            "Technology": "technology",
            "method": "method",
            "output_raw": "output_raw",
            "start_date": "start_date",
            "operator": "operator_raw",
        },
        "tier": 1,
    },
    "egat_power_plants": {
        "path": CLEANED_DIR / "egat_power_plants.geojson",
        "column_map": {
            "name": "name",
            "name_th": "name_th",
            "capacity_mw": "capacity_mw",
            "status": "status",
        },
        "tier": 1,
    },
    "egat_substations": {
        "path": CLEANED_DIR / "egat_substations.geojson",
        "column_map": {
            "name": "name",
            "name_th": "name_th",
            "code": "code",
            "province": "province",
            "district": "district",
        },
        "tier": 1,
    },
    "egat_gen_data": {
        "path": CLEANED_DIR / "egat_gen_data.geojson",
        "column_map": {
            "name": "name",
            "name_th": "name_th",
            "capacity_mw": "capacity_mw",
            "status": "status",
        },
        "tier": 1,
    },
    "koh_samui_grid": {
        "path": CLEANED_DIR / "koh_samui_grid_infrastructure.geojson",
        "column_map": {
            "name": "name",
            "type": "type",
            "voltage": "voltage",
            "operator": "operator",
            "osmid": "osmid",
        },
        "tier": 1,
    },
    "spotlight_khanom": {
        "path": GEOJSON_DIR / "spotlight-Khanom-power-station-103km.geojson",
        "column_map": {
            "type": "type",
            "name": "name",
            "radius_km": "radius_km",
        },
        "tier": 1,
    },
    "thailand_hv_grid": {
        "path": GEOJSON_DIR / "thailand_hv_grid_sample.geojson",
        "column_map": {
            "osmid": "osmid",
            "power": "power",
            "voltage": "voltage",
            "operator": "operator",
            "name": "name",
        },
        "tier": 1,
    },

    # ── Tier 2: Transmission infrastructure (moderate size) ──────────────────
    "egat_lines": {
        "path": CLEANED_DIR / "egat_lines.geojson",
        "column_map": {
            "name": "name",
            "code": "code",
            "voltage_kv": "voltage_kv",
            "status": "status",
            "fid": "fid",
        },
        "tier": 2,
    },
    "egat_towers": {
        "path": CLEANED_DIR / "egat_combined_towers.geojson",
        "column_map": {
            "tower_number": "tower_number",
            "line_code": "line_code",
            "line_name": "line_name",
            "fid": "fid",
        },
        "tier": 2,
    },

    # ── Tier 3: Heavy datasets (loaded only with --all) ──────────────────────
    "egat_combined_gen": {
        "path": CLEANED_DIR / "egat_combined_gen.geojson",
        "column_map": {
            "name": "name",
            "name_th": "name_th",
            "capacity_mw": "capacity_mw",
            "status": "status",
        },
        "tier": 3,
    },
    "egat_combined_load": {
        "path": CLEANED_DIR / "egat_combined_load.geojson",
        "column_map": {
            "name": "name",
            "name_th": "name_th",
            "province": "province",
            "load_mw": "load_mw",
            "fid": "fid",
        },
        "tier": 3,
    },
    "egat_district_load": {
        "path": CLEANED_DIR / "egat_district_load.geojson",
        "column_map": {
            "name": "name",
            "name_th": "name_th",
            "province": "province",
            "load_mw": "load_mw",
        },
        "tier": 3,
    },
    "egat_gen_zones": {
        "path": CLEANED_DIR / "egat_gen_zones.geojson",
        "column_map": {
            "name": "name",
            "name_th": "name_th",
            "province": "province",
            "load_mw": "load_mw",
        },
        "tier": 3,
    },
    "pea_hv_conductor_zones": {
        "path": GEOJSON_DIR / "pea_hvcond_merge.geojson",
        "column_map": {
            "zone": "zone",
            "egat_sub_id": "egat_sub_id",
            "egat_subname": "egat_subname",
            "objectid": "objectid",
        },
        "tier": 3,
    },
    "pea_mv_conductor_zones": {
        "path": CLEANED_DIR / "pea_nohv_mvcond_merge.geojson",
        "column_map": {
            "name": "name",
            "name_th": "name_th",
            "province": "province",
            "load_mw": "load_mw",
        },
        "tier": 3,
    },
}


def get_engine(url: str):
    """Create SQLAlchemy engine and ensure PostGIS extension exists."""
    engine = create_engine(url, echo=False)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        conn.commit()
    return engine


def detect_geom_type(gdf: gpd.GeoDataFrame) -> str:
    """Detect the PostGIS geometry type from a GeoDataFrame."""
    geom_types = gdf.geometry.geom_type.unique().tolist()
    if len(geom_types) == 1:
        return GEOJSON_TYPE_MAP.get(geom_types[0], "GEOMETRY")
    # Mixed types — use generic GEOMETRY
    return "GEOMETRY"


def build_table(table_name: str, columns: dict, geom_type: str, metadata: MetaData) -> Table:
    """Dynamically build a SQLAlchemy Table from column definitions."""
    cols = [Column("id", Integer, primary_key=True, autoincrement=True)]

    for col_name in columns.values():
        if col_name in ("capacity_mw", "load_mw", "radius_km"):
            cols.append(Column(col_name, Float, nullable=True))
        else:
            cols.append(Column(col_name, String, nullable=True))

    # Use GEOGRAPHY for accurate distance queries on global/country-wide data
    cols.append(Column("geom", Geography(geom_type, srid=4326)))

    return Table(table_name, metadata, *cols)


def load_and_rename(path: Path, column_map: dict) -> gpd.GeoDataFrame:
    """Read GeoJSON, rename columns, and keep only mapped ones + geometry."""
    gdf = gpd.read_file(path)

    if gdf.empty:
        return gdf

    # Drop exact duplicates (spatial + attributes)
    initial_count = len(gdf)
    gdf = gdf.drop_duplicates()
    new_count = len(gdf)
    if initial_count > new_count:
        print(f"   ✂️  Dropped {initial_count - new_count} duplicate rows")

    # Rename columns per map
    rename = {k: v for k, v in column_map.items() if k in gdf.columns}
    gdf = gdf.rename(columns=rename)

    # Keep only mapped columns + geometry
    keep = [v for v in column_map.values() if v in gdf.columns] + ["geometry"]
    gdf = gdf[[c for c in keep if c in gdf.columns]]

    # Ensure CRS is WGS84
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)

    return gdf


def ingest_table(engine, table_name: str, gdf: gpd.GeoDataFrame, column_map: dict) -> int:
    """Create table and bulk-insert a GeoDataFrame.

    Uses direct SQLAlchemy insert with EWKT strings instead of
    gdf.to_postgis() because the latter calls find_srid() which
    fails for GEOGRAPHY columns.
    """
    metadata = MetaData()
    geom_type = detect_geom_type(gdf)
    table = build_table(table_name, column_map, geom_type, metadata)

    # Drop and recreate
    metadata.drop_all(engine, tables=[table], checkfirst=True)
    metadata.create_all(engine)

    # Prepare rows
    rows = []
    attr_cols = [c for c in gdf.columns if c != "geometry"]
    for _, row in gdf.iterrows():
        record = {col: row[col] for col in attr_cols}
        # Convert geometry to EWKT (Extended WKT with SRID)
        if row.geometry is not None:
            record["geom"] = f"SRID=4326;{row.geometry.wkt}"
        else:
            record["geom"] = None
        rows.append(record)

    # Bulk insert in batches of 500 to handle large datasets
    batch_size = 500
    with engine.begin() as conn:
        for i in range(0, len(rows), batch_size):
            conn.execute(table.insert(), rows[i:i + batch_size])

    # Create spatial index
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_geom
            ON {table_name} USING GIST (geom)
        """))

    # Return count
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()


def print_summary(engine):
    """Print summary of all spatial tables in the database."""
    insp = inspect(engine)
    tables = sorted(insp.get_table_names())

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  PostGIS Database Summary — gridtokenx_geo                         ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")

    total_rows = 0
    # Use a fresh connection for each table to avoid transaction abort lockouts
    for tbl in tables:
        if tbl in ("spatial_ref_sys", "geography_columns", "geometry_columns", "addr"):
            continue
            
        try:
            with engine.connect() as conn:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
                total_rows += count

                # Get geometry type
                geom_info = "?"
                try:
                    # Check if 'geom' column exists first
                    columns = [c["name"] for c in insp.get_columns(tbl)]
                    if "geom" in columns:
                        geom_info = conn.execute(text(f"""
                            SELECT GeometryType(geom::geometry)
                            FROM {tbl} LIMIT 1
                        """)).scalar()
                except Exception:
                    pass

                has_idx = 0
                try:
                    has_idx = conn.execute(text(f"""
                        SELECT COUNT(*) FROM pg_indexes
                        WHERE tablename = '{tbl}' AND indexname LIKE '%geom%'
                    """)).scalar()
                except Exception:
                    pass

                idx_mark = "✓" if has_idx else "✗"
                print(f"║  {tbl:<30s}  {count:>7,d} rows  {geom_info or '?':<18s}  idx:{idx_mark}  ║")
        except Exception as e:
            print(f"║  {tbl:<30s}  [Error: {str(e)[:40]}] {' ' * 13} ║")

    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Total: {len([t for t in tables if t not in ('spatial_ref_sys', 'geography_columns', 'geometry_columns', 'addr')])} tables, {total_rows:,d} rows{' ' * 35}║")

    # Ko Tao proximity check across power plant tables
    ko_tao_q = """
        SELECT 'power_plants' AS source, plant_name AS name, type, capacity_mw,
               ROUND((ST_Distance(geom, ST_SetSRID(ST_MakePoint(99.84, 10.08), 4326)::geography) / 1000)::numeric, 1) AS dist_km
        FROM power_plants
        WHERE ST_DWithin(geom, ST_SetSRID(ST_MakePoint(99.84, 10.08), 4326)::geography, 100000)
        UNION ALL
        SELECT 'koh_samui_grid', name, type, NULL,
               ROUND((ST_Distance(geom, ST_SetSRID(ST_MakePoint(99.84, 10.08), 4326)::geography) / 1000)::numeric, 1)
        FROM koh_samui_grid
        WHERE ST_DWithin(geom, ST_SetSRID(ST_MakePoint(99.84, 10.08), 4326)::geography, 100000)
        ORDER BY dist_km
    """
    try:
        with engine.connect() as conn:
            nearby = conn.execute(text(ko_tao_q)).fetchall()
        print("╠══════════════════════════════════════════════════════════════════════╣")
        print(f"║  Grid assets within 100 km of Ko Tao: {len(nearby):<28d} ║")
        for row in nearby[:15]:
            cap = f"{row.capacity_mw:.0f}MW" if row.capacity_mw else ""
            print(f"║    {row.dist_km:6.1f}km  {row.source:<16s}  {(row.type or ''):<12s} {cap:<8s} {(row.name or 'Unknown')[:20]}")
    except Exception:
        pass  # Tables may not exist if only a subset was loaded

    print("╚══════════════════════════════════════════════════════════════════════╝")


def main():
    parser = argparse.ArgumentParser(
        description="Load GeoJSON spatial data into PostGIS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tier system:
  Tier 1 — Core grid assets (< 1 MB each, always loaded)
  Tier 2 — Transmission infrastructure (< 5 MB, always loaded)
  Tier 3 — Heavy datasets (60-270 MB cleaned, loaded with --all)

Examples:
  python data/load_postgis.py                          # Load tier 1+2
  python data/load_postgis.py --all                    # Load all tiers
  python data/load_postgis.py --tables power_plants    # Load specific table
  python data/load_postgis.py --tables egat_lines,egat_towers  # Multiple
        """,
    )
    parser.add_argument(
        "--url",
        default=os.getenv("POSTGIS_URL", DEFAULT_URL),
        help="PostGIS connection URL",
    )
    parser.add_argument(
        "--tables",
        type=str,
        default=None,
        help="Comma-separated list of specific tables to load",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Load all tiers including heavy datasets (tier 3)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_tables",
        help="List available tables and exit",
    )
    args = parser.parse_args()

    # List mode
    if args.list_tables:
        print(f"\n{'Table':<30s} {'Tier':>4s} {'File':<50s}")
        print("-" * 90)
        for name, cfg in sorted(TABLE_REGISTRY.items(), key=lambda x: (x[1]["tier"], x[0])):
            exists = "✓" if cfg["path"].exists() else "✗"
            size = f"{cfg['path'].stat().st_size / 1e6:.1f}MB" if cfg["path"].exists() else "missing"
            print(f"  {name:<28s} {cfg['tier']:>4d}   {exists} {cfg['path'].name:<40s} {size}")
        return

    # Determine which tables to load
    if args.tables:
        selected = [t.strip() for t in args.tables.split(",")]
        for t in selected:
            if t not in TABLE_REGISTRY:
                print(f"❌ Unknown table: {t}")
                print(f"   Available: {', '.join(sorted(TABLE_REGISTRY.keys()))}")
                sys.exit(1)
    elif args.all:
        selected = list(TABLE_REGISTRY.keys())
    else:
        # Default: tier 1 + 2
        selected = [name for name, cfg in TABLE_REGISTRY.items() if cfg["tier"] <= 2]

    print(f"🗄️  PostGIS: {args.url.split('@')[1] if '@' in args.url else args.url}")
    print(f"📋 Loading {len(selected)} tables: {', '.join(selected)}\n")

    # Connect
    engine = get_engine(args.url)

    # Ingest each table
    results = []
    for table_name in selected:
        cfg = TABLE_REGISTRY[table_name]
        path = cfg["path"]

        if not path.exists():
            print(f"⚠️  {table_name}: file not found ({path.name}), skipping")
            results.append((table_name, 0, 0, "SKIPPED"))
            continue

        size_mb = path.stat().st_size / 1e6
        print(f"📂 {table_name} ({path.name}, {size_mb:.1f} MB)")

        t0 = time.time()
        try:
            gdf = load_and_rename(path, cfg["column_map"])
            count = ingest_table(engine, table_name, gdf, cfg["column_map"])
            elapsed = time.time() - t0
            print(f"   ✅ {count:,d} rows inserted in {elapsed:.1f}s\n")
            results.append((table_name, count, elapsed, "OK"))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"   ❌ Error: {e}\n")
            results.append((table_name, 0, elapsed, str(e)[:60]))

    # Results summary
    print("\n" + "=" * 70)
    print(f"{'Table':<30s} {'Rows':>8s} {'Time':>7s} {'Status':<12s}")
    print("-" * 70)
    for name, count, elapsed, status in results:
        print(f"  {name:<28s} {count:>8,d} {elapsed:>6.1f}s {status:<12s}")
    print("=" * 70)

    ok_count = sum(1 for _, _, _, s in results if s == "OK")
    total_rows = sum(r[1] for r in results)
    print(f"\n✅ {ok_count}/{len(results)} tables loaded, {total_rows:,d} total rows\n")

    # Full database summary
    print_summary(engine)


if __name__ == "__main__":
    main()
