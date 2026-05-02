"use client";

import { useState, useEffect, Suspense } from 'react';

export default function MapPage() {
  const [MapComponent, setMapComponent] = useState<React.ComponentType | null>(null);

  useEffect(() => {
    import('./map-content').then(mod => {
      setMapComponent(() => mod.default);
    });
  }, []);

  if (!MapComponent) {
    return (
      <div className="h-screen w-screen bg-slate-950 flex items-center justify-center">
        <div className="text-slate-400">Loading map...</div>
      </div>
    );
  }

  return (
    <Suspense fallback={
      <div className="h-screen w-screen bg-slate-950 flex items-center justify-center">
        <div className="text-slate-400">Initializing map context...</div>
      </div>
    }>
      <MapComponent />
    </Suspense>
  );
}
