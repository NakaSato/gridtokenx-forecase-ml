"use client";

import { useRef, useState } from 'react';
import Map, { Source, Layer, NavigationControl, type MapRef } from 'react-map-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { MAPBOX_TOKEN } from '@/lib/mapbox';
import { useGridAssets } from '@/hooks/useGridAssets';
import { useMapStyle } from '@/hooks/useMapStyle';
import { Database, Loader2, RefreshCw } from 'lucide-react';

export function PostGisMap() {
    const [table, setTable] = useState('egat_power_plants');
    const { data, loading, error, refresh } = useGridAssets(table, 500);
    const { mapStyle } = useMapStyle();
    const [viewState, setViewState] = useState({
        longitude: 100.5,
        latitude: 13.7,
        zoom: 5
    });

    const tables = [
        { id: 'egat_power_plants', label: 'EGAT Power Plants' },
        { id: 'egat_substations', label: 'EGAT Substations' },
        { id: 'egat_lines', label: 'EGAT Transmission Lines' },
        { id: 'koh_samui_grid', label: 'Koh Samui Grid' },
        { id: 'power_plants', label: 'All Power Plants' }
    ];

    return (
        <div className="relative w-full h-full bg-slate-950 rounded-xl overflow-hidden border border-slate-800">
            {/* Header / Controls */}
            <div className="absolute top-4 left-4 z-10 space-y-2">
                <div className="bg-slate-900/90 backdrop-blur-md p-3 rounded-lg border border-slate-700 shadow-xl">
                    <div className="flex items-center gap-2 mb-3">
                        <Database className="w-5 h-5 text-cyan-400" />
                        <h3 className="text-white font-bold">PostGIS Spatial Data</h3>
                    </div>
                    
                    <div className="space-y-1">
                        {tables.map(t => (
                            <button
                                key={t.id}
                                onClick={() => setTable(t.id)}
                                className={`w-full text-left px-3 py-1.5 rounded text-xs font-medium transition ${
                                    table === t.id 
                                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30' 
                                    : 'text-slate-400 hover:bg-slate-800 hover:text-white'
                                }`}
                            >
                                {t.label}
                            </button>
                        ))}
                    </div>

                    <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between">
                        <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">
                            {loading ? 'Fetching...' : `${data?.features?.length || 0} Assets`}
                        </div>
                        <button 
                            onClick={refresh}
                            className="text-slate-400 hover:text-white transition"
                            disabled={loading}
                        >
                            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
                        </button>
                    </div>
                </div>
            </div>

            {error && (
                <div className="absolute top-4 right-4 z-10 bg-red-500/20 border border-red-500/50 text-red-200 px-4 py-2 rounded-lg text-xs backdrop-blur-md">
                    Error: {error}
                </div>
            )}

            <Map
                {...viewState}
                onMove={evt => setViewState(evt.viewState)}
                mapboxAccessToken={MAPBOX_TOKEN}
                mapStyle={mapStyle}
                style={{ width: '100%', height: '100%' }}
            >
                <NavigationControl position="bottom-right" />

                {data && (
                    <Source id="postgis-data" type="geojson" data={data}>
                        {/* Line Layer */}
                        <Layer
                            id="postgis-lines"
                            type="line"
                            filter={['==', ['$geometryType'], 'LineString']}
                            paint={{
                                'line-color': '#06b6d4',
                                'line-width': 2,
                                'line-opacity': 0.8
                            }}
                        />
                        
                        {/* Point Layer */}
                        <Layer
                            id="postgis-points"
                            type="circle"
                            filter={['==', ['$geometryType'], 'Point']}
                            paint={{
                                'circle-radius': 5,
                                'circle-color': table.includes('plant') ? '#f59e0b' : '#3b82f6',
                                'circle-stroke-width': 1,
                                'circle-stroke-color': '#fff',
                                'circle-opacity': 0.8
                            }}
                        />

                        {/* Polygon Layer */}
                        <Layer
                            id="postgis-polygons"
                            type="fill"
                            filter={['==', ['$geometryType'], 'Polygon']}
                            paint={{
                                'fill-color': '#8b5cf6',
                                'fill-opacity': 0.2,
                                'fill-outline-color': '#8b5cf6'
                            }}
                        />
                    </Source>
                )}
            </Map>
        </div>
    );
}
