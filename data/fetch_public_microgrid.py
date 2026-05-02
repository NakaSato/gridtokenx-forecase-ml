"""
Download freely available microgrid datasets (no login required).
"""
import requests
from pathlib import Path

OUT = Path(__file__).parent / "public_datasets"
OUT.mkdir(exist_ok=True)


def download(url: str, filename: str, stream=True):
    dest = OUT / filename
    if dest.exists():
        print(f"  skip (exists): {filename}")
        return
    print(f"  downloading: {filename} ...", end=" ", flush=True)
    r = requests.get(url, stream=stream, timeout=180)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"{dest.stat().st_size / 1e6:.2f} MB")


DATASETS = [
    # 1. Microgrid Energy System (GitHub) — hourly load, PV, market price, carbon intensity
    (
        "https://raw.githubusercontent.com/FLYao123/Dataset-2---A-Microgrid-Energy-System/main/Weather_factors_and_material_all.csv",
        "microgrid_weather_material.csv",
    ),
    (
        "https://raw.githubusercontent.com/FLYao123/Dataset-2---A-Microgrid-Energy-System/main/new_buy_sale_2012_Initial_data.csv",
        "microgrid_load_price_pv.csv",
    ),
    # 2. Zenodo 7602546 — Energy community: PV + BESS + EV (250 households, hourly)
    (
        "https://zenodo.org/api/records/7602546/files/EC_EV_dataset.xlsx/content",
        "zenodo_pv_bess_ev_community.xlsx",
    ),
    # 3. Open Energy Data Initiative (OEDI) — NREL commercial building load profiles
    (
        "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1/metadata/metadata.parquet",
        "nrel_resstock_metadata.parquet",
    ),
]

if __name__ == "__main__":
    print(f"Saving to: {OUT}\n")
    for url, fname in DATASETS:
        try:
            download(url, fname)
        except Exception as e:
            print(f"  FAILED: {fname} — {e}")
    print(f"\nDone. Files in: {OUT}")
    print("\nDataset summary:")
    print("  microgrid_weather_material.csv  — hourly weather + material costs")
    print("  microgrid_load_price_pv.csv     — hourly load, buy/sell price, PV gen")
    print("  zenodo_pv_bess_ev_community.xlsx — PV + BESS + EV community (250 HH)")
    print("  nrel_resstock_metadata.parquet  — NREL building stock metadata")
