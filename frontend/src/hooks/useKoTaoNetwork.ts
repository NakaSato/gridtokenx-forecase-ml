"use client";

/**
 * Ko Tao Network GeoJSON Hook
 *
 * Fetches the static /ko_tao_network.geojson and splits features
 * by `properties.type` into separate GeoJSON FeatureCollections
 * for map rendering.
 *
 * Feature types in the file:
 *   - spotlight_boundary  → coverage area polygon (~58 km radius)
 *   - spotlight_center    → hub center point (970 MW)
 *   - plant               → Khanom Power Station
 *   - substation           → 115 kV substations (×6)
 *   - transmission         → 115/230 kV lines (LineString + MultiLineString)
 */

import { useState, useEffect, useMemo } from "react";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface KoTaoSubstation {
  name: string;
  distance_km: number;
  substation_type: string | null;
  voltage_kv: number | null;
  source: string;
  coordinates: [number, number];
}

interface KoTaoPlant {
  name: string;
  distance_km: number;
  capacity_mw: number;
  technology: string | null;
  status: string;
  source: string;
  coordinates: [number, number];
}

interface KoTaoTransmission {
  name: string;
  distance_km: number;
  voltage_class: string; // e.g. "115000", "230000"
  voltage_kv: number;    // derived: voltage_class / 1000
  source: string;
}

export interface KoTaoNetworkData {
  /** Spotlight boundary polygon */
  boundary: GeoJSON.FeatureCollection;
  /** Spotlight center point (hub) */
  center: GeoJSON.FeatureCollection;
  /** Substations */
  substations: GeoJSON.FeatureCollection;
  /** Parsed substation metadata */
  substationList: KoTaoSubstation[];
  /** Power plant(s) */
  plants: GeoJSON.FeatureCollection;
  /** Parsed plant metadata */
  plantList: KoTaoPlant[];
  /** Transmission lines (LineString + MultiLineString) */
  transmission: GeoJSON.FeatureCollection;
  /** Parsed transmission metadata */
  transmissionList: KoTaoTransmission[];
  /** Total generation capacity in MW */
  totalCapacityMw: number;
}

/* ------------------------------------------------------------------ */
/*  Empty collection helper                                            */
/* ------------------------------------------------------------------ */

const EMPTY_FC: GeoJSON.FeatureCollection = {
  type: "FeatureCollection",
  features: [],
};

/* ------------------------------------------------------------------ */
/*  Hook                                                               */
/* ------------------------------------------------------------------ */

export function useKoTaoNetwork() {
  const [raw, setRaw] = useState<GeoJSON.FeatureCollection | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch once
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const res = await fetch("/ko_tao_network.geojson");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (!cancelled) setRaw(json);
      } catch (err) {
        if (!cancelled) {
          console.error("[useKoTaoNetwork] fetch error:", err);
          setError(err instanceof Error ? err.message : "Failed to load");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // Split by feature type
  const data = useMemo<KoTaoNetworkData | null>(() => {
    if (!raw?.features) return null;

    const boundary: GeoJSON.Feature[] = [];
    const center: GeoJSON.Feature[] = [];
    const substations: GeoJSON.Feature[] = [];
    const plants: GeoJSON.Feature[] = [];
    const transmission: GeoJSON.Feature[] = [];

    const substationList: KoTaoSubstation[] = [];
    const plantList: KoTaoPlant[] = [];
    const transmissionList: KoTaoTransmission[] = [];
    let totalCapacityMw = 0;

    for (const feature of raw.features) {
      const props = feature.properties || {};
      const geomType = feature.geometry?.type;

      switch (props.type) {
        case "spotlight_boundary":
          boundary.push(feature);
          break;

        case "spotlight_center":
          center.push(feature);
          totalCapacityMw += props.generation_mw || 0;
          break;

        case "plant":
          plants.push(feature);
          if (geomType === "Point") {
            const coords = (feature.geometry as GeoJSON.Point).coordinates;
            plantList.push({
              name: props.name,
              distance_km: props.distance_km,
              capacity_mw: props.capacity_mw || 0,
              technology: props.technology,
              status: props.status,
              source: props.source,
              coordinates: coords as [number, number],
            });
          }
          break;

        case "substation":
          substations.push(feature);
          if (geomType === "Point") {
            const coords = (feature.geometry as GeoJSON.Point).coordinates;
            substationList.push({
              name: props.name,
              distance_km: props.distance_km,
              substation_type: props.substation_type,
              voltage_kv: props.voltage_kv,
              source: props.source,
              coordinates: coords as [number, number],
            });
          }
          break;

        case "transmission": {
          // Enrich with derived voltage_kv for styling
          const voltageKv = props.voltage_class
            ? Math.round(Number(props.voltage_class) / 1000)
            : 0;
          const enriched: GeoJSON.Feature = {
            ...feature,
            properties: { ...props, voltage_kv: voltageKv },
          };
          transmission.push(enriched);
          transmissionList.push({
            name: props.name,
            distance_km: props.distance_km,
            voltage_class: props.voltage_class,
            voltage_kv: voltageKv,
            source: props.source,
          });
          break;
        }
      }
    }

    return {
      boundary: { type: "FeatureCollection", features: boundary },
      center: { type: "FeatureCollection", features: center },
      substations: { type: "FeatureCollection", features: substations },
      substationList,
      plants: { type: "FeatureCollection", features: plants },
      plantList,
      transmission: { type: "FeatureCollection", features: transmission },
      transmissionList,
      totalCapacityMw,
    };
  }, [raw]);

  return {
    data,
    loading,
    error,
    // Convenience accessors (safe even when data is null)
    boundary: data?.boundary ?? EMPTY_FC,
    center: data?.center ?? EMPTY_FC,
    substations: data?.substations ?? EMPTY_FC,
    plants: data?.plants ?? EMPTY_FC,
    transmission: data?.transmission ?? EMPTY_FC,
  };
}
