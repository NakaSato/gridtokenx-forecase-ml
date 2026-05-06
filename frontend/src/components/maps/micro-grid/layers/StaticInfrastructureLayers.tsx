import React from 'react';
import { Source, Layer } from 'react-map-gl';

interface StaticInfrastructureLayersProps {
    showInfraData?: boolean;
}

export const StaticInfrastructureLayers: React.FC<StaticInfrastructureLayersProps> = ({ 
    showInfraData = true 
}) => {
    return (
        <>
            {/* Additional Infrastructure from OpenInfraMap (Base Geography/Assets) */}
            <Source id="openinframap-data" type="geojson" data="/samui_infra.geojson" />
            
            {showInfraData && (
                <>
                    <Layer
                        id="infra-landuse"
                        source="openinframap-data"
                        type="fill"
                        filter={['==', ['get', 'vt_layer'], 'landuse']}
                        paint={{
                            'fill-color': [
                                'match',
                                ['get', 'kind'],
                                ['wood', 'forest', 'grass', 'park', 'nature_reserve'], '#14532d', // Deep Green
                                ['water', 'ocean', 'river', 'stream', 'canal', 'bay'], '#0c4a6e', // Deep Blue
                                ['beach', 'sand'], '#78350f', // Sand/Brown
                                ['residential', 'retail', 'commercial', 'industrial'], '#1e1b4b', // Dark Purple/Blue
                                ['marina', 'pier'], '#164e63', // Cyan-Blue
                                '#1e293b' // Default Slate
                            ],
                            'fill-opacity': 0.1
                        }}
                    />
                    <Layer
                        id="infra-roads"
                        source="openinframap-data"
                        type="line"
                        filter={['==', ['get', 'vt_layer'], 'roads']}
                        paint={{
                            'line-color': '#475569',
                            'line-width': 1,
                            'line-opacity': 0.1
                        }}
                    />
                    <Layer
                        id="infra-buildings"
                        source="openinframap-data"
                        type="fill"
                        filter={['==', ['get', 'vt_layer'], 'buildings']}
                        paint={{
                            'fill-color': '#94a3b8',
                            'fill-opacity': 0.1
                        }}
                    />
                </>
            )}

            {/* Consolidated Power Infrastructure from OSM/OpenInfraMap */}
            <Source id="power-infra-grid" type="geojson" data="/power_infrastructure.geojson" />
            
            {/* General Infrastructure (Togglable) */}
            {showInfraData && (
                <>
                    <Layer
                        id="power-areas"
                        source="power-infra-grid"
                        type="fill"
                        filter={['all', 
                            ['match', ['geometry-type'], ['Polygon', 'MultiPolygon'], true, false],
                            ['!=', ['get', 'type'], 'spotlight_boundary'],
                            ['!=', ['get', 'is_master'], true]
                        ]}
                        paint={{
                            'fill-color': [
                                'match',
                                ['get', 'power'],
                                'substation', '#10b981',
                                'plant', '#f43f5e',
                                '#3b82f6'
                            ],
                            'fill-opacity': 0.4,
                            'fill-outline-color': '#fff'
                        }}
                    />
                    <Layer
                        id="power-lines"
                        source="power-infra-grid"
                        type="line"
                        filter={['all',
                            ['match', ['get', 'power'], ['line', 'minor_line', 'cable', 'catenary_mast'], true, false],
                            ['!=', ['get', 'is_master'], true]
                        ]}
                        paint={{
                            'line-color': [
                                'coalesce',
                                ['get', 'stroke'],
                                [
                                    'interpolate',
                                    ['linear'],
                                    ['coalesce', ['get', 'voltage'], 0],
                                    0, '#94a3b8',
                                    22, '#10b981',
                                    33, '#06b6d4',
                                    115, '#3b82f6',
                                    230, '#f59e0b',
                                    500, '#dc2626'
                                ]
                            ],
                            'line-width': [
                                'interpolate',
                                ['linear'],
                                ['zoom'],
                                0, 1.5,
                                10, 2,
                                14, 3
                            ],
                            'line-opacity': 0.4
                        }}
                    />
                    <Layer
                        id="power-nodes"
                        source="power-infra-grid"
                        type="circle"
                        filter={['all',
                            ['match', ['get', 'power'], ['substation', 'plant', 'generator', 'transformer', 'pole'], true, false],
                            ['!=', ['get', 'is_master'], true]
                        ]}
                        paint={{
                            'circle-radius': [
                                'match',
                                ['get', 'power'],
                                'substation', 4,
                                'plant', 6,
                                'pole', 1.5,
                                2
                            ],
                            'circle-color': [
                                'match',
                                ['get', 'power'],
                                'substation', '#10b981',
                                'plant', '#f43f5e',
                                'pole', '#64748b',
                                '#3b82f6'
                            ],
                            'circle-opacity': 0.6
                        }}
                    />
                </>
            )}

            {/* Master Grid (High-Fidelity Orchestration Layer) - Always Visible */}
            <Layer
                id="master-tx-glow"
                source="power-infra-grid"
                type="line"
                filter={['all', ['==', ['get', 'type'], 'transmission'], ['==', ['get', 'is_master'], true]]}
                paint={{
                    'line-color': [
                        'coalesce',
                        ['get', 'stroke'],
                        [
                            'match',
                            ['get', 'voltage_class'],
                            '500000', '#dc2626',
                            '230000', '#f59e0b',
                            '115000', '#3b82f6',
                            '33000', '#06b6d4',
                            '22000', '#10b981',
                            '#3b82f6'
                        ]
                    ],
                    'line-width': 6,
                    'line-blur': 4,
                    'line-opacity': 0.2
                }}
            />
            <Layer
                id="master-tx-line"
                source="power-infra-grid"
                type="line"
                filter={['all', ['==', ['get', 'type'], 'transmission'], ['==', ['get', 'is_master'], true]]}
                paint={{
                    'line-color': [
                        'coalesce',
                        ['get', 'stroke'],
                        [
                            'match',
                            ['get', 'voltage_class'],
                            '500000', '#dc2626',
                            '230000', '#f59e0b',
                            '115000', '#3b82f6',
                            '33000', '#06b6d4',
                            '22000', '#10b981',
                            '#3b82f6'
                        ]
                    ],
                    'line-width': [
                        'interpolate',
                        ['linear'],
                        ['zoom'],
                        8, [
                            'match', 
                            ['get', 'voltage_class'], 
                            '500000', 3,
                            '230000', 2.5,
                            '115000', 2,
                            1.5
                        ],
                        14, [
                            'match', 
                            ['get', 'voltage_class'], 
                            '500000', 5,
                            '230000', 4,
                            '115000', 3,
                            2.5
                        ]
                    ],
                    'line-opacity': 1.0,
                    'line-dasharray': [
                        'match',
                        ['get', 'circuit_id'],
                        'KMB', ["literal", [4, 4]],
                        'KMA', ["literal", [3, 2]],
                        ["literal", [1]]
                    ]
                }}
            />
            <Layer
                id="master-nodes"
                source="power-infra-grid"
                type="circle"
                filter={['all', ['match', ['get', 'type'], ['substation', 'emergency_generation', 'spinning_reserve', 'plant'], true, false], ['==', ['get', 'is_master'], true]]}
                paint={{
                    'circle-radius': [
                        'interpolate',
                        ['linear'],
                        ['zoom'],
                        8, ['match', ['get', 'type'], 'plant', 4, 3],
                        14, ['match', ['get', 'type'], 'plant', 8, 5]
                    ],
                    'circle-color': [
                        'match',
                        ['get', 'type'],
                        'substation', '#10b981',
                        'emergency_generation', '#f59e0b',
                        'spinning_reserve', '#f43f5e',
                        'plant', '#f43f5e',
                        '#3b82f6'
                    ],
                    'circle-stroke-width': 0,
                    'circle-stroke-color': '#fff'
                }}
            />
            <Layer
                id="master-node-labels"
                source="power-infra-grid"
                type="symbol"
                filter={['all', ['match', ['get', 'type'], ['substation', 'emergency_generation', 'spinning_reserve', 'plant'], true, false], ['==', ['get', 'is_master'], true]]}
                layout={{
                    'text-field': [
                        'case',
                        ['has', 'id'],
                        ['concat', '#', ['get', 'id'], ' ', ['get', 'name']],
                        ['get', 'name']
                    ],
                    'text-size': [
                        'interpolate',
                        ['linear'],
                        ['zoom'],
                        8, 10,
                        14, 12
                    ],
                    'text-font': ['Open Sans Bold', 'Arial Unicode MS Bold'],
                    'text-variable-anchor': ['top', 'bottom', 'left', 'right'],
                    'text-radial-offset': 1.0,
                    'text-justify': 'auto'
                }}
                paint={{
                    'text-color': '#fff',
                    'text-halo-color': 'rgba(0,0,0,0.8)',
                    'text-halo-width': 1.5,
                    'text-opacity': [
                        'interpolate',
                        ['linear'],
                        ['zoom'],
                        8, 0.2,
                        10, 1
                    ]
                }}
            />
            <Layer
                id="master-tx-labels"
                source="power-infra-grid"
                type="symbol"
                filter={['all', ['==', ['get', 'type'], 'transmission'], ['==', ['get', 'is_master'], true]]}
                layout={{
                    'text-field': ['coalesce', ['get', 'label'], ['get', 'name']],
                    'symbol-placement': 'line',
                    'text-size': 13,
                    'text-font': ['Open Sans Regular', 'Arial Unicode MS Regular'],
                    'text-letter-spacing': 0.1,
                    'text-max-angle': 30,
                    'text-rotation-alignment': 'map',
                    'text-pitch-alignment': 'viewport'
                }}
                paint={{
                    'text-color': '#93c5fd', // Light blue
                    'text-halo-color': 'rgba(0,0,0,0.8)',
                    'text-halo-width': 1.5,
                    'text-opacity': [
                        'interpolate',
                        ['linear'],
                        ['zoom'],
                        8, 0,
                        9, 1
                    ]
                }}
            />

            {/* Spotlight Boundary (Search Radius) */}
            <Layer
                id="spotlight-boundary"
                source="power-infra-grid"
                type="line"
                filter={['==', ['get', 'type'], 'spotlight_boundary']}
                paint={{
                    'line-color': '#94a3b8',
                    'line-width': 1.5,
                    'line-dasharray': [4, 4],
                    'line-opacity': 0.4
                }}
            />
        </>
    );
};
