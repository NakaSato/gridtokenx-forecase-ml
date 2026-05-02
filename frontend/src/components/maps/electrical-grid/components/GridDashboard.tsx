import React from 'react';
import {
  Activity,
  ShieldCheck,
  AlertTriangle,
  Layers,
  Database,
  BarChart3,
  Cpu
} from 'lucide-react';
import { ComprehensiveGridStats } from '../hooks/useGridStats';

interface GridDashboardProps {
  stats: ComprehensiveGridStats;
  visible: boolean;
}

export const GridDashboard: React.FC<GridDashboardProps> = ({ stats, visible }) => {
  if (!visible || !stats.status) return null;

  const isHealthy = stats.status.status === 'running';

  return (
    <div className="absolute top-20 left-4 z-[1000] w-72 pointer-events-none">
      <div className="flex flex-col gap-3">



      </div>
    </div>
  );
};
