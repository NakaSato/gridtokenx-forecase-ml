import { NextRequest } from 'next/server';
import { proxyGET } from '../proxy-utils';

export async function GET(request: NextRequest) {
    const { searchParams } = new URL(request.url);
    const table = searchParams.get('table') || 'egat_power_plants';
    const limit = searchParams.get('limit') || '100';
    
    return proxyGET(`/grid/assets?table=${table}&limit=${limit}`);
}
