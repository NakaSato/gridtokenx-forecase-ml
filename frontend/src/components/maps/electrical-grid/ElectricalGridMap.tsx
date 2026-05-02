"use client";

/**
 * Electrical Grid Map Component
 * 
 * Interactive map showing Thai electrical infrastructure (EGAT, MEA, PEA)
 */

import React, { useState, useCallback, useMemo, useRef } from 'react';
import { useKoTaoNetwork } from '@/hooks/useKoTaoNetwork';
import { usePersistedViewState } from '@/hooks/usePersistedViewState';
import Map, {
  Source,
  Layer,
  NavigationControl,
  ScaleControl,
  GeolocateControl,
  FullscreenControl,
  Popup
} from 'react-map-gl';
import type {
  ViewStateChangeEvent
} from 'react-map-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { Zap, Globe, Moon, Satellite, RefreshCw, Loader2, CircuitBoard, MapPin, Network } from 'lucide-react';
import { useNetwork } from '@/components/providers/NetworkProvider';
import { useMapStyle } from '@/hooks/useMapStyle';
import type {
  ElectricalInfrastructure,
  ElectricalGridFeatureCollection,
  FilterState
} from './types';
import {
  DEFAULT_FILTERS
} from './types';
import {
  getSubstationCircleLayer,
  getPoleCircleLayer,
  getSubstationGlowLayer,
  getTransmissionLineLayer,
  getTransmissionLineGlowLayer
} from './layers/electricalGridLayers';
import { FilterPanel } from './components/FilterPanel';
import { InfrastructurePopup } from './components/InfrastructurePopup';
import { MapLegend } from './components/MapLegend';
import { useElectricalGridData } from './hooks/useElectricalGridData';
import { useEgatTransmissionData } from './hooks/useEgatTransmissionData';
import { useGridStats } from './hooks/useGridStats';
import { GridDashboard } from './components/GridDashboard';


const ElectricalGridMap = () => {
  const { getApiUrl } = useNetwork();

  // State
  const [viewState, setViewState] = usePersistedViewState('electrical-grid', {
    longitude: 99.99007762999207,
    latitude: 9.528326082141575,
    zoom: 6
  });
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const [selectedInfrastructure, setSelectedInfrastructure] = useState<ElectricalInfrastructure | null>(null);
  const [popupInfo, setPopupInfo] = useState<{
    longitude: number;
    latitude: number;
    feature: ElectricalInfrastructure;
  } | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [showLegend, setShowLegend] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { mapStyle, toggle, isSatellite } = useMapStyle();

  // Ko Tao Network overlay (opt-in, default off for infra view)
  const [showKoTaoOverlay, setShowKoTaoOverlay] = useState(false);
  const koTao = useKoTaoNetwork();

  // Data fetching
  const { infrastructure, stats: infraStats, loading: infraLoading, lastRefresh: infraLastRefresh, refresh: refreshInfra } = useElectricalGridData(getApiUrl);
  const { data: egatData, loading: egatLoading, lastRefresh: egatLastRefresh, refresh: refreshEgat } = useEgatTransmissionData(getApiUrl);
  const gridStats = useGridStats(getApiUrl);

  const loading = infraLoading || egatLoading;
  const lastRefresh = infraLastRefresh || egatLastRefresh;

  const refreshData = useCallback(() => {
    refreshInfra();
    refreshEgat();
    gridStats.refresh();
  }, [refreshInfra, refreshEgat, gridStats]);

  // Refs
  const mapRef = useRef<any>(null);

  // Filter infrastructure based on current filters
  const filteredInfrastructure = infrastructure.filter(item => {
    if (!filters.operators.includes(item.operator)) return false;
    if (!filters.types.includes(item.type)) return false;
    if (filters.voltageLevels.length > 0 && item.voltage_kv && !filters.voltageLevels.includes(item.voltage_kv)) return false;
    if (filters.provinces.length > 0 && item.province && !filters.provinces.includes(item.province)) return false;
    if (filters.searchQuery && !searchMatch(item, filters.searchQuery)) return false;
    return true;
  });

  // Convert to GeoJSON for Points
  const pointGeoJsonData: ElectricalGridFeatureCollection = {
    type: 'FeatureCollection',
    features: filteredInfrastructure.map(item => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [item.longitude, item.latitude]
      },
      properties: item
    }))
  };

  // Convert to GeoJSON for Lines (EGAT)
  const lineGeoJsonData = useMemo(() => {
    if (!egatData || !filters.operators.includes('EGAT')) return null;

    return {
      type: 'FeatureCollection' as const,
      features: egatData.lines
        .filter(l => filters.voltageLevels.length === 0 || filters.voltageLevels.includes(l.voltage_kv))
        .map(l => {
          const fromSub = egatData.substations.find(s => s.id === l.from);
          const toSub = egatData.substations.find(s => s.id === l.to);

          if (!fromSub || !toSub) return null;

          return {
            type: 'Feature' as const,
            geometry: {
              type: 'LineString' as const,
              coordinates: [
                [fromSub.longitude, fromSub.latitude],
                [toSub.longitude, toSub.latitude]
              ]
            },
            properties: l
          };
        })
        .filter((f): f is any => f !== null)
    };
  }, [egatData, filters.operators, filters.voltageLevels]);

  // Search match helper
  const searchMatch = (item: ElectricalInfrastructure, query: string): boolean => {
    const searchStr = query.toLowerCase();
    return Boolean(
      item.name_en?.toLowerCase().includes(searchStr) ||
      item.name_th?.toLowerCase().includes(searchStr) ||
      item.id.toLowerCase().includes(searchStr) ||
      item.province?.toLowerCase().includes(searchStr) ||
      item.district?.toLowerCase().includes(searchStr)
    );
  };

  // Handle map click
  const handleMapClick = useCallback((event: any) => {
    const features = event.features;
    if (features && features.length > 0) {
      const feature = features[0].properties as ElectricalInfrastructure;
      setPopupInfo({
        longitude: event.lngLat.lng,
        latitude: event.lngLat.lat,
        feature
      });
    } else {
      setPopupInfo(null);
    }
  }, []);

  // Handle view change
  const handleViewStateChange = useCallback((event: ViewStateChangeEvent) => {
    setViewState(event.viewState);
  }, []);

  // Toggle layer visibility
  const toggleLayer = useCallback((layerId: string) => {
    // Implementation for toggling individual layers
    console.log('Toggle layer:', layerId);
  }, []);

  // Update filters
  const updateFilters = useCallback((updates: Partial<FilterState>) => {
    setFilters(prev => ({ ...prev, ...updates }));
  }, []);

  // Reset filters
  const resetFilters = useCallback(() => {
    setFilters(DEFAULT_FILTERS);
  }, []);

  // Fit bounds to show all infrastructure
  const fitToInfrastructure = useCallback(() => {
    if (mapRef.current && filteredInfrastructure.length > 0) {
      const bounds = filteredInfrastructure.reduce(
        (acc, item) => {
          acc.west = Math.min(acc.west, item.longitude);
          acc.east = Math.max(acc.east, item.longitude);
          acc.south = Math.min(acc.south, item.latitude);
          acc.north = Math.max(acc.north, item.latitude);
          return acc;
        },
        { west: 180, east: -180, south: 90, north: -90 }
      );

      mapRef.current.fitBounds(
        [[bounds.west, bounds.south], [bounds.east, bounds.north]],
        { padding: 50, duration: 1000 }
      );
    }
  }, [filteredInfrastructure]);

  // Loading state
  if (loading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-gray-900">
        <div className="text-center">
          <Zap className="w-16 h-16 text-yellow-400 mx-auto mb-4 animate-pulse" />
          <h2 className="text-2xl font-bold text-white mb-2">Loading Electrical Grid</h2>
          <p className="text-gray-400">Fetching infrastructure data from EGAT, MEA, and PEA...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-gray-900">
        <div className="text-center max-w-md">
          <h2 className="text-2xl font-bold text-red-400 mb-4">Error Loading Map</h2>
          <p className="text-gray-400 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-yellow-500 text-black rounded-lg hover:bg-yellow-600"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen relative bg-gray-900">
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-[1000] bg-gradient-to-b from-black/80 to-transparent p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-amber-400" />
            <h2 className="text-white font-bold text-lg">Grid Infrastructure</h2>
            <span className="text-gray-400 text-sm">EGAT / MEA / PEA</span>
          </div>
          <div className="flex items-center gap-2">
            {lastRefresh && (
              <span className="text-gray-500 text-xs">
                Updated {lastRefresh.toLocaleTimeString()}
              </span>
            )}
            <button
              onClick={refreshData}
              disabled={loading}
              className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white disabled:opacity-50 transition"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
            </button>
            <button
              onClick={() => setShowKoTaoOverlay(!showKoTaoOverlay)}
              className={`p-1.5 rounded-lg transition flex items-center gap-1 text-xs font-bold ${
                showKoTaoOverlay
                  ? 'bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30'
                  : 'bg-white/10 text-white/50 hover:bg-white/20 hover:text-white'
              }`}
              title="Toggle Ko Tao Network overlay"
            >
              <Network className="w-4 h-4" />
              <span className="hidden sm:inline">Ko Tao</span>
            </button>
            <button
              onClick={toggle}
              className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white transition"
            >
              {isSatellite ? <MapPin className="w-4 h-4" /> : <CircuitBoard className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>

      {/* Filter Panel */}
      {showFilters && (
        <div className="absolute top-16 right-4 sm:top-20 sm:right-6 z-[1000] w-80 max-h-[80vh] overflow-y-auto">
          <FilterPanel
            filters={filters}
            onUpdateFilters={updateFilters}
            onResetFilters={resetFilters}
            onClose={() => setShowFilters(false)}
            stats={infraStats}
          />
        </div>
      )}

      {/* Grid Dashboard */}
      <GridDashboard stats={gridStats} visible={true} />

      {/* Legend */}
      {showLegend && (
        <MapLegend
          visible={showLegend}
          onClose={() => setShowLegend(false)}
        />
      )}

      {/* Map */}
      <div className="relative w-full h-full">
        <button
          onClick={toggle}
          className="absolute top-16 right-4 z-[1000] bg-slate-900/80 backdrop-blur-xl border border-white/10 rounded-lg px-2.5 py-1.5 text-xs text-slate-400 hover:text-white transition-colors"
          title="Toggle satellite view"
        >
          <Globe className="w-3.5 h-3.5 inline mr-1" /> {isSatellite ? <Satellite className="w-3.5 h-3.5 inline" /> : <Moon className="w-3.5 h-3.5 inline" />}
        </button>
        <Map
          ref={mapRef}
          {...viewState}
          onMove={handleViewStateChange}
          onClick={handleMapClick}
          style={{ width: '100%', height: '100%' }}
          mapStyle={mapStyle}
          projection={{ name: 'globe' }}
          interactiveLayerIds={[
            'egat-substations',
            'mea-substations',
            'pea-substations',
            'egat-poles',
            'mea-poles',
            'pea-poles',
            'transmission-lines-overhead',
            'transmission-lines-submarine',
            'transmission-lines-underground'
          ]}
        >
          {/* Controls */}
          <NavigationControl position="top-right" />
          <GeolocateControl position="top-right" />
          <FullscreenControl position="top-right" />
          <ScaleControl position="bottom-left" />

          {/* Ko Tao Network overlay (background context) */}
          {showKoTaoOverlay && koTao.data && (
            <>
              <Source id="kotao-eg-boundary" type="geojson" data={koTao.boundary}>
                <Layer
                  id="kotao-eg-boundary-outline"
                  type="line"
                  paint={{
                    'line-color': '#06b6d4',
                    'line-width': 1.5,
                    'line-opacity': 0.3,
                    'line-dasharray': [4, 3],
                  }}
                />
                <Layer
                  id="kotao-eg-boundary-fill"
                  type="fill"
                  paint={{
                    'fill-color': '#06b6d4',
                    'fill-opacity': 0.03,
                  }}
                />
              </Source>
              <Source id="kotao-eg-transmission" type="geojson" data={koTao.transmission}>
                <Layer
                  id="kotao-eg-tx-glow"
                  type="line"
                  paint={{
                    'line-color': [
                      'match',
                      ['get', 'voltage_kv'],
                      230, '#f59e0b',
                      115, '#3b82f6',
                      '#6b7280',
                    ],
                    'line-width': 5,
                    'line-opacity': 0.1,
                    'line-blur': 4,
                  }}
                />
                <Layer
                  id="kotao-eg-tx-line"
                  type="line"
                  paint={{
                    'line-color': [
                      'match',
                      ['get', 'voltage_kv'],
                      230, '#f59e0b',
                      115, '#3b82f6',
                      '#6b7280',
                    ],
                    'line-width': [
                      'match',
                      ['get', 'voltage_kv'],
                      230, 2.5,
                      115, 1.5,
                      1,
                    ],
                    'line-opacity': 0.7,
                  }}
                />
              </Source>
              <Source id="kotao-eg-plants" type="geojson" data={koTao.plants}>
                <Layer
                  id="kotao-eg-plant-dot"
                  type="circle"
                  paint={{
                    'circle-radius': 7,
                    'circle-color': '#f59e0b',
                    'circle-stroke-width': 2,
                    'circle-stroke-color': '#fff',
                    'circle-opacity': 0.9,
                  }}
                />
                <Layer
                  id="kotao-eg-plant-label"
                  type="symbol"
                  layout={{
                    'text-field': ['concat', ['get', 'name'], '\n', ['to-string', ['get', 'capacity_mw']], ' MW'],
                    'text-size': 10,
                    'text-offset': [0, 1.8],
                    'text-anchor': 'top',
                    'text-allow-overlap': false,
                  }}
                  paint={{
                    'text-color': '#fbbf24',
                    'text-halo-color': '#000',
                    'text-halo-width': 1.5,
                  }}
                />
              </Source>
              <Source id="kotao-eg-substations" type="geojson" data={koTao.substations}>
                <Layer
                  id="kotao-eg-sub-dot"
                  type="circle"
                  paint={{
                    'circle-radius': 5,
                    'circle-color': '#3b82f6',
                    'circle-stroke-width': 1.5,
                    'circle-stroke-color': '#000',
                    'circle-opacity': 0.8,
                  }}
                />
              </Source>
            </>
          )}

          {/* Transmission Line layers */}
          {lineGeoJsonData && (
            <Source id="egat-lines" type="geojson" data={lineGeoJsonData as any}>
              {/* Glow layers first */}
              <Layer {...getTransmissionLineGlowLayer('overhead')} />
              <Layer {...getTransmissionLineGlowLayer('submarine')} />
              <Layer {...getTransmissionLineGlowLayer('underground')} />
              
              {/* Primary layers */}
              <Layer {...getTransmissionLineLayer('overhead')} />
              <Layer {...getTransmissionLineLayer('submarine')} />
              <Layer {...getTransmissionLineLayer('underground')} />
            </Source>
          )}

          {/* Infrastructure point layers */}
          <Source id="electrical-infrastructure" type="geojson" data={pointGeoJsonData}>
            {/* EGAT Substations */}
            <Layer {...getSubstationGlowLayer('EGAT')} />
            <Layer {...getSubstationCircleLayer('EGAT')} />

            {/* MEA Substations */}
            <Layer {...getSubstationGlowLayer('MEA')} />
            <Layer {...getSubstationCircleLayer('MEA')} />

            {/* PEA Substations */}
            <Layer {...getSubstationGlowLayer('PEA')} />
            <Layer {...getSubstationCircleLayer('PEA')} />

            {/* EGAT Poles/Towers */}
            <Layer {...getPoleCircleLayer('EGAT')} />

            {/* MEA Poles */}
            <Layer {...getPoleCircleLayer('MEA')} />

            {/* PEA Poles */}
            <Layer {...getPoleCircleLayer('PEA')} />
          </Source>

          {/* Popup */}
          {popupInfo && (
            <Popup
              longitude={popupInfo.longitude}
              latitude={popupInfo.latitude}
              anchor="bottom"
              closeButton={true}
              onCloseClick={() => setPopupInfo(null)}
              closeOnClick={false}
              className="electrical-grid-popup"
            >
              <InfrastructurePopup
                infrastructure={popupInfo.feature}
                onSelect={setSelectedInfrastructure}
              />
            </Popup>
          )}
        </Map>
      </div>

      {/* Stats Bar (Hidden in favor of GridDashboard) */}
      {/* <div className="absolute bottom-4 left-4 bg-gray-800/90 backdrop-blur-sm rounded-lg px-3 py-2 flex items-center gap-3 text-white z-[1000]">
        ...
      </div> */}
    </div>
  );
};

export default ElectricalGridMap;
