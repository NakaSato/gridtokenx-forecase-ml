// @ts-ignore - mapbox-gl/dist/mapbox-gl.js has no types
import mapboxgl from 'mapbox-gl';

// Next.js env var (NEXT_PUBLIC_ prefix required for client access)
const raw = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || '';
export const MAPBOX_TOKEN = raw.trim().replace(/^"|"$/g, '');

// Set token globally (required by mapbox-gl-js v2)
if (MAPBOX_TOKEN && MAPBOX_TOKEN !== 'your_mapbox_token_here' && MAPBOX_TOKEN !== 'your_token_here') {
    mapboxgl.accessToken = MAPBOX_TOKEN;
}

export { mapboxgl };
