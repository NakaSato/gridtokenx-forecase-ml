"""
Download NREL ARPA-E PERFORM dataset (no login required).
Fetches: Load actuals, Solar actuals, Wind actuals for ERCOT 2018.
Files are HDF5 (.h5) — hourly time-series with probabilistic forecasts.
"""
import requests
from pathlib import Path

BASE = "https://arpa-e-perform.s3.us-west-2.amazonaws.com"
OUT = Path(__file__).parent / "nrel_perform"
OUT.mkdir(exist_ok=True)

FILES = [
    "ERCOT/2018/Load/Actuals/BA_level/BA_load_actuals_2018.h5",
    "ERCOT/2018/Solar/Actuals/BA_level/BA_solar_actuals_2018.h5",
    "ERCOT/2018/Wind/Actuals/BA_level/BA_wind_actuals_2018.h5",
    "ERCOT/MetaData/load_meta.xlsx",
    "ERCOT/MetaData/solar_meta.xlsx",
    "ERCOT/MetaData/wind_meta.xlsx",
]


def download(key: str):
    url = f"{BASE}/{key}"
    dest = OUT / Path(key).name
    if dest.exists():
        print(f"  skip (exists): {dest.name}")
        return
    print(f"  downloading: {dest.name} ...", end=" ", flush=True)
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"{dest.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    print(f"Saving to: {OUT}\n")
    for key in FILES:
        download(key)
    print("\nDone. Load with: pd.read_hdf('data/raw/nrel_perform/<file>.h5')")
