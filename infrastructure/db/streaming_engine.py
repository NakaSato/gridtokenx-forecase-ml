import sqlite3
import numpy as np
import math
from collections import deque
from typing import List, Optional
from domain.entities import TelemetryRow
from domain.grid import IslandGrid
from domain.interfaces import IStreamingEngine

class SQLiteStreamingEngine(IStreamingEngine):
    def __init__(self, window_size: int, db_path: str, cfg: dict, seq_fields: List[str]):
        self.window_size = window_size
        self.db_path = db_path
        self.cfg = cfg
        self.seq_fields = seq_fields
        self.buffer: deque = deque(maxlen=window_size)
        self._actuals:   List[float] = []
        self._forecasts: List[float] = []
        
        # Initialize physical grid state
        self.grid = IslandGrid("Ko Tao", cfg)
        
        from domain.entities import TelemetryRow
        self.all_fields = list(TelemetryRow.model_fields.keys())
        
        self._init_db()
        self._load_state()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cols = ", ".join([f"{f} REAL" for f in self.all_fields])
            conn.execute(f"CREATE TABLE IF NOT EXISTS telemetry (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
            conn.execute("CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, actual REAL, forecast REAL)")
            conn.commit()

    def _load_state(self):
        from domain.entities import TelemetryRow
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM telemetry")
            if cursor.fetchone()[0] > 0:
                rows = conn.execute(f"SELECT {', '.join(self.all_fields)} FROM (SELECT * FROM telemetry ORDER BY id DESC LIMIT ?) ORDER BY id ASC", (self.window_size,)).fetchall()
                for r in rows:
                    row_obj = TelemetryRow(**dict(zip(self.all_fields, r)))
                    self.buffer.append(row_obj)
                    # Sync grid state
                    self.grid.update(row_obj.tao_load_mw, self.cfg["data"]["circuit_cap_max"] - row_obj.headroom_mw)

            metrics = conn.execute("SELECT actual, forecast FROM metrics ORDER BY id ASC").fetchall()
            for a, f in metrics:
                self._actuals.append(a)
                self._forecasts.append(f)
        print(f"   [Infrastructure] Restored {len(self.buffer)} telemetry rows and {len(self._actuals)} metric pairs.")

    def ingest(self, row: TelemetryRow, circuit_cap_mw: Optional[float] = None):
        self.buffer.append(row)
        
        # Update physical grid simulation state
        cap = circuit_cap_mw if circuit_cap_mw is not None else (row.headroom_mw + row.tao_load_mw)
        self.grid.update(row.tao_load_mw, cap)

        with sqlite3.connect(self.db_path) as conn:
            vals = [getattr(row, f) for f in self.all_fields]
            placeholders = ", ".join(["?"] * len(self.all_fields))
            conn.execute(f"INSERT INTO telemetry ({', '.join(self.all_fields)}) VALUES ({placeholders})", vals)
            conn.execute("DELETE FROM telemetry WHERE id NOT IN (SELECT id FROM telemetry ORDER BY id DESC LIMIT 2000)")
            conn.commit()

    def is_ready(self) -> bool:
        return len(self.buffer) == self.window_size

    def record_actual(self, actual: float, forecast: float):
        self._actuals.append(actual)
        self._forecasts.append(forecast)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO metrics (actual, forecast) VALUES (?, ?)", (actual, forecast))
            conn.commit()

    def live_metrics(self) -> dict:
        n = len(self._actuals)
        grid_status = self.grid.get_status()
        
        if n == 0:
            return {
                "n": 0, "mae": None, "rmse": None, "mape": None,
                "grid_status": grid_status
            }
        
        a = np.array(self._actuals)
        f = np.array(self._forecasts)
        err = a - f
        mae  = float(np.mean(np.abs(err)))
        rmse = float(math.sqrt(np.mean(err ** 2)))
        mape = float(np.mean(np.abs(err / (a + 1e-8))) * 100)
        
        return {
            "n": n, 
            "mae": round(mae, 4), 
            "rmse": round(rmse, 4), 
            "mape": round(mape, 4),
            "error_history": err[-24:].tolist(),
            "grid_status": grid_status
        }
