import { useState, useEffect, useCallback } from 'react';

export interface GridTopology {
  buses: number;
  lines: number;
  trafos: number;
  loads: number;
  sgens: number;
}

export interface GridStatus {
  status: string;
  meters_online: number;
  grid_frequency_hz: number;
}

export interface EgatStats {
  total_substations: number;
  total_transmission_lines: number;
  total_line_length_km: number;
  substations_500kv: number;
  line_length_500kv_km: number;
  total_capacity_500kv_mva: number;
  substations_230kv: number;
  line_length_230kv_km: number;
  total_capacity_230kv_mva: number;
  substations_115kv: number;
  line_length_115kv_km: number;
  total_capacity_115kv_mva: number;
  regions_covered: string[];
  provinces_covered: string[];
}

export interface PowerPlantStats {
  total: { count: number; capacity_mw: number };
  renewable: { capacity_mw: number; percentage: number };
  by_type: Record<string, { plant_count: number; total_capacity_mw: number }>;
}

export interface ComprehensiveGridStats {
  topology: GridTopology | null;
  status: GridStatus | null;
  egat: EgatStats | null;
  plants: PowerPlantStats | null;
  loading: boolean;
  lastUpdated: Date | null;
}

export const useGridStats = (getApiUrl: (path: string) => string) => {
  const [data, setData] = useState<ComprehensiveGridStats>({
    topology: null,
    status: null,
    egat: null,
    plants: null,
    loading: true,
    lastUpdated: null,
  });

  const fetchAllStats = useCallback(async () => {
    try {
      const [topoRes, statusRes, egatRes, plantRes] = await Promise.all([
        fetch(getApiUrl('/api/v1/grid/topology')),
        fetch(getApiUrl('/api/v1/grid/status')),
        fetch(getApiUrl('/api/v1/grid/egat/statistics')),
        fetch(getApiUrl('/api/v1/power-plants/stats'))
      ]);

      const [topoData, statusData, egatData, plantData] = await Promise.all([
        topoRes.ok ? topoRes.json() : null,
        statusRes.ok ? statusRes.json() : null,
        egatRes.ok ? egatRes.json() : null,
        plantRes.ok ? plantRes.json() : null
      ]);

      setData({
        topology: topoData?.topology || null,
        status: statusData || null,
        egat: egatData || null,
        plants: plantData || null,
        loading: false,
        lastUpdated: new Date()
      });
    } catch (error) {
      console.error("Failed to fetch comprehensive grid stats:", error);
      setData(prev => ({ ...prev, loading: false }));
    }
  }, [getApiUrl]);

  useEffect(() => {
    fetchAllStats();
    const interval = setInterval(fetchAllStats, 10000);
    return () => clearInterval(interval);
  }, [fetchAllStats]);

  return { ...data, refresh: fetchAllStats };
};
