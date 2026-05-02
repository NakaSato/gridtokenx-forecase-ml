"use client";

import { useState } from 'react';
import { Activity, AlertTriangle, CheckCircle, Zap } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';

// ── Types ─────────────────────────────────────────────────────────────────────

interface ForecastResult {
  forecast_mw: number[];
  summary: {
    peak_load_mw: number;
    min_load_mw: number;
    avg_load_mw: number;
    total_fuel_saved_liters: number;
    bess_cycles: number;
  };
  device: string;
}

interface MetricsResult {
  mape: number;
  mae: number;
  r2: number;
  fuel_savings_pct: number;
}

interface Warning {
  level: string;
  message: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const HOURS = Array.from({ length: 24 }, (_, i) => `${String(i).padStart(2, '0')}:00`);

function parseCSV(raw: string): number[] {
  return raw.split(',').map((v) => parseFloat(v.trim())).filter((v) => !isNaN(v));
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function ForecastPage() {
  const [circuitCSV, setCircuitCSV] = useState(
    '8.1,8.3,8.0,7.9,8.2,8.5,9.1,9.8,10.2,10.5,10.3,10.1,9.9,9.7,9.5,9.8,10.1,10.4,10.2,9.8,9.3,8.9,8.5,8.2'
  );
  const [initialSoc, setInitialSoc] = useState(0.65);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [forecast, setForecast] = useState<ForecastResult | null>(null);
  const [metrics, setMetrics] = useState<MetricsResult | null>(null);
  const [warnings, setWarnings] = useState<Warning[]>([]);

  async function fetchMetrics() {
    try {
      const res = await fetch('/api/gridtokenx/metrics');
      if (res.ok) setMetrics(await res.json());
    } catch { /* non-critical */ }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setForecast(null);
    setWarnings([]);

    const circuit = parseCSV(circuitCSV);
    if (circuit.length !== 24) {
      setError('Circuit forecast must have exactly 24 values.');
      return;
    }

    setLoading(true);
    try {
      // We need a 168-row history window — use the /stream/metrics health check
      // to confirm the API is up, then call /forecast with dummy history
      const healthRes = await fetch('/api/gridtokenx/health');
      if (!healthRes.ok) throw new Error('API offline');
      const health = await healthRes.json();
      const windowSize: number = health.window ?? 168;

      // Build minimal history rows (repeat last circuit value as load proxy)
      const avgLoad = circuit.reduce((a, b) => a + b, 0) / 24;
      const historyRow = {
        island_load_mw: avgLoad,
        load_lag_1h: avgLoad,
        load_lag_24h: avgLoad,
        bess_soc_pct: initialSoc * 100,
        dry_bulb_temp: 32.0,
        heat_index: 38.0,
        rel_humidity: 75.0,
        hour_of_day: 12.0,
        is_high_season: 1.0,
      };

      const lgbm_features: Record<string, number> = {
        island_load_mw: avgLoad,
        load_lag_1h: avgLoad,
        load_lag_24h: avgLoad,
        bess_soc_pct: initialSoc * 100,
        dry_bulb_temp: 32.0,
        heat_index: 38.0,
        rel_humidity: 75.0,
        hour_of_day: 12.0,
        is_high_season: 1.0,
        month: new Date().getMonth() + 1,
        day_of_week: new Date().getDay(),
        is_weekend: new Date().getDay() >= 5 ? 1 : 0,
        load_lag_168h: avgLoad,
        load_roll_mean_24h: avgLoad,
        load_roll_std_24h: 0.2,
        load_roll_mean_168h: avgLoad,
        heat_index_lag_1h: 38.0,
        heat_index_roll_mean_24h: 38.0,
        tourist_index: 0.8,
        solar_hour: 0.5,
        peak_flag: 0.0,
      };

      const body = JSON.stringify({
        history: Array(windowSize).fill(historyRow),
        circuit_forecast: circuit,
        initial_soc: initialSoc,
        lgbm_features,
      });

      const fcRes = await fetch('/api/gridtokenx/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      });
      if (!fcRes.ok) {
        const err = await fcRes.json();
        throw new Error(err.detail ?? JSON.stringify(err));
      }
      const fcData: ForecastResult = await fcRes.json();
      setForecast(fcData);

      // Fetch warnings
      const warnRes = await fetch('/api/gridtokenx/warnings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          load_forecast: fcData.forecast_mw,
          circuit_forecast: circuit,
          current_soc: initialSoc,
          lookahead_hours: 6,
        }),
      });
      if (warnRes.ok) {
        const warnData = await warnRes.json();
        setWarnings(warnData.warnings ?? []);
      }

      fetchMetrics();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const chartData = forecast?.forecast_mw.map((mw, i) => ({
    hour: HOURS[i],
    forecast: parseFloat(mw.toFixed(3)),
    circuit: parseCSV(circuitCSV)[i] ?? null,
  })) ?? [];

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6">
      <div className="max-w-5xl mx-auto space-y-6">

        {/* Header */}
        <div className="flex items-center gap-3">
          <Zap className="text-yellow-400" size={28} />
          <div>
            <h1 className="text-2xl font-bold">24h Load Forecast</h1>
            <p className="text-gray-400 text-sm">Ko Tao–Phangan–Samui Hybrid TCN-LGBM</p>
          </div>
        </div>

        {/* Model metrics banner */}
        {metrics && (
          <div className="grid grid-cols-4 gap-3">
            {[
              { label: 'MAPE', value: `${metrics.mape?.toFixed(2)}%` },
              { label: 'MAE', value: `${metrics.mae?.toFixed(3)} MW` },
              { label: 'R²', value: metrics.r2?.toFixed(4) },
              { label: 'Fuel Saved', value: `${metrics.fuel_savings_pct?.toFixed(1)}%` },
            ].map(({ label, value }) => (
              <div key={label} className="bg-gray-900 rounded-lg p-3 text-center">
                <div className="text-xs text-gray-400">{label}</div>
                <div className="text-lg font-semibold text-yellow-400">{value}</div>
              </div>
            ))}
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="bg-gray-900 rounded-xl p-5 space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">
              Circuit Forecast — 24 comma-separated MW values
            </label>
            <textarea
              className="w-full bg-gray-800 rounded-lg p-3 text-sm font-mono text-gray-100 border border-gray-700 focus:outline-none focus:border-yellow-500 resize-none"
              rows={3}
              value={circuitCSV}
              onChange={(e) => setCircuitCSV(e.target.value)}
            />
          </div>
          <div className="flex items-center gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Initial BESS SoC</label>
              <input
                type="number"
                min={0.2} max={0.95} step={0.05}
                value={initialSoc}
                onChange={(e) => setInitialSoc(parseFloat(e.target.value))}
                className="w-28 bg-gray-800 rounded-lg p-2 text-sm border border-gray-700 focus:outline-none focus:border-yellow-500"
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="mt-5 px-6 py-2 bg-yellow-500 hover:bg-yellow-400 disabled:opacity-50 text-gray-900 font-semibold rounded-lg transition-colors"
            >
              {loading ? 'Running…' : 'Run Forecast'}
            </button>
          </div>
          {error && (
            <div className="flex items-center gap-2 text-red-400 text-sm">
              <AlertTriangle size={16} /> {error}
            </div>
          )}
        </form>

        {/* Results */}
        {forecast && (
          <>
            {/* Summary cards */}
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: 'Peak Load', value: `${forecast.summary.peak_load_mw?.toFixed(2)} MW` },
                { label: 'Avg Load', value: `${forecast.summary.avg_load_mw?.toFixed(2)} MW` },
                { label: 'Fuel Saved', value: `${forecast.summary.total_fuel_saved_liters?.toFixed(0)} L` },
              ].map(({ label, value }) => (
                <div key={label} className="bg-gray-900 rounded-lg p-4">
                  <div className="text-xs text-gray-400">{label}</div>
                  <div className="text-xl font-bold text-white">{value}</div>
                </div>
              ))}
            </div>

            {/* Chart */}
            <div className="bg-gray-900 rounded-xl p-5">
              <div className="flex items-center gap-2 mb-4">
                <Activity size={18} className="text-yellow-400" />
                <span className="font-semibold">Forecast vs Circuit ({forecast.device})</span>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="hour" tick={{ fontSize: 11, fill: '#9ca3af' }} interval={3} />
                  <YAxis tick={{ fontSize: 11, fill: '#9ca3af' }} unit=" MW" />
                  <Tooltip
                    contentStyle={{ background: '#111827', border: '1px solid #374151' }}
                    labelStyle={{ color: '#f9fafb' }}
                  />
                  <ReferenceLine y={forecast.summary.avg_load_mw} stroke="#6b7280" strokeDasharray="4 4" />
                  <Line type="monotone" dataKey="forecast" stroke="#eab308" strokeWidth={2} dot={false} name="Hybrid Forecast" />
                  <Line type="monotone" dataKey="circuit" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="Circuit" strokeDasharray="5 3" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Warnings */}
            {warnings.length > 0 && (
              <div className="bg-gray-900 rounded-xl p-5 space-y-2">
                <div className="font-semibold text-sm text-gray-300 mb-2">Early Warnings</div>
                {warnings.map((w, i) => (
                  <div key={i} className={`flex items-start gap-2 text-sm p-2 rounded-lg ${w.level === 'CRITICAL' ? 'bg-red-950 text-red-300' : 'bg-yellow-950 text-yellow-300'}`}>
                    <AlertTriangle size={15} className="mt-0.5 shrink-0" />
                    <span><span className="font-semibold">{w.level}</span> — {w.message}</span>
                  </div>
                ))}
              </div>
            )}
            {warnings.length === 0 && (
              <div className="flex items-center gap-2 text-green-400 text-sm">
                <CheckCircle size={16} /> No warnings for the next 6 hours.
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
