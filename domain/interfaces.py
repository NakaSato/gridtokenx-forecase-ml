from abc import ABC, abstractmethod
from typing import List, Optional, Dict
import numpy as np
from domain.entities import TelemetryRow, HourlyDispatch

class IPredictor(ABC):
    @abstractmethod
    def predict(self, history: List[TelemetryRow], lgbm_features: dict) -> List[float]:
        pass

class IDispatchOptimizer(ABC):
    @abstractmethod
    def run_dispatch(self, load_forecast: np.ndarray, circuit_forecast: np.ndarray, 
                     initial_soc: float, cfg: dict) -> List[HourlyDispatch]:
        pass

class IStreamingEngine(ABC):
    @abstractmethod
    def ingest(self, row: TelemetryRow, circuit_cap_mw: Optional[float] = None) -> None:
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        pass

    @abstractmethod
    def record_actual(self, actual: float, forecast: float) -> None:
        pass

    @abstractmethod
    def live_metrics(self) -> dict:
        pass
