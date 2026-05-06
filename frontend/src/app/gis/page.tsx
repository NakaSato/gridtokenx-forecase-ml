import { PostGisMap } from '@/components/maps/PostGisMap';
import Link from 'next/link';
import { ArrowLeft, Globe } from 'lucide-react';

export default function GisPage() {
    return (
        <main className="flex flex-col h-screen bg-slate-950">
            {/* Header */}
            <header className="flex items-center justify-between p-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur-md">
                <div className="flex items-center gap-4">
                    <Link 
                        href="/dashboard" 
                        className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition text-slate-400 hover:text-white"
                    >
                        <ArrowLeft className="w-5 h-5" />
                    </Link>
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-cyan-500/20 rounded-lg">
                            <Globe className="w-6 h-6 text-cyan-400" />
                        </div>
                        <div>
                            <h1 className="text-xl font-bold text-white">Spatial Intelligence</h1>
                            <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">PostGIS Asset Explorer</p>
                        </div>
                    </div>
                </div>
                
                <div className="flex items-center gap-4 px-4 py-2 bg-slate-800/50 rounded-full border border-slate-700/50">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                        <span className="text-[10px] font-bold text-slate-300 uppercase">Live Database</span>
                    </div>
                </div>
            </header>

            {/* Map Container */}
            <div className="flex-1 p-4 overflow-hidden">
                <PostGisMap />
            </div>

            {/* Footer Status */}
            <footer className="px-6 py-2 border-t border-slate-900 bg-slate-950 text-[10px] text-slate-600 flex justify-between">
                <div>GRIDTOKENX SPATIAL ENGINE v2.0</div>
                <div className="font-mono">SRID: 4326 (WGS84) • PROXIMITY_LIMIT: 100KM</div>
            </footer>
        </main>
    );
}
