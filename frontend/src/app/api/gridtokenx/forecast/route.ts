import { NextRequest } from 'next/server';
import { proxyPOST } from '../proxy-utils';
export async function POST(req: NextRequest) {
    return proxyPOST('/forecast', await req.text());
}
