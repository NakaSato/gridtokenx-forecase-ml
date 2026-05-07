"use client";

import React, { useState, useEffect } from 'react';
import { Bot, ChevronRight, MessageSquare, ShieldAlert, Zap, ArrowRight, Info } from 'lucide-react';
import { cn } from '@/lib/common';

interface GemmaAgentProps {
  type: 'forecast' | 'dispatch' | 'incident';
  data: any;
  className?: string;
}

export const GemmaAgent: React.FC<GemmaAgentProps> = ({ type, data, className }) => {
  const [insight, setInsight] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [expanded, setExpanded] = useState<boolean>(true);

  const fetchInsight = async () => {
    if (!data) return;
    setLoading(true);
    try {
      let endpoint = '';
      let body = {};

      if (type === 'forecast') {
        endpoint = '/api/gridtokenx/agent/forecast-narrative';
        body = { forecast_mw: data.forecast_mw, lgbm_features: data.lgbm_features };
      } else if (type === 'dispatch') {
        endpoint = '/api/gridtokenx/agent/explain-dispatch';
        body = { optimized_schedule: data.optimized, baseline_schedule: data.baseline };
      } else if (type === 'incident') {
        endpoint = '/api/gridtokenx/agent/action-plan';
        body = { incident: data };
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (res.ok) {
        const result = await res.json();
        setInsight(result.explanation || result.narrative || result.action_plan || '');
      }
    } catch (err) {
      console.error('Gemma Agent Error:', err);
      setInsight('ระบบ Gemma ไม่สามารถดึงข้อมูลเชิงลึกได้ในขณะนี้');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchInsight();
  }, [type, data]);

  if (!data && !loading) return null;

  return (
    <div className={cn(
      "relative overflow-hidden rounded-2xl transition-all duration-500",
      "bg-slate-900/40 border border-slate-800/50 backdrop-blur-xl shadow-2xl",
      expanded ? "p-6" : "p-4 cursor-pointer hover:bg-slate-800/60",
      className
    )}
    onClick={() => !expanded && setExpanded(true)}
    >
      {/* Background Glow */}
      <div className="absolute -top-24 -right-24 w-48 h-48 bg-emerald-500/10 blur-[80px] rounded-full pointer-events-none" />
      <div className="absolute -bottom-24 -left-24 w-48 h-48 bg-blue-500/10 blur-[80px] rounded-full pointer-events-none" />

      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Bot className="w-6 h-6 text-white" />
            </div>
            {loading && (
              <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-emerald-500 rounded-full border-2 border-slate-900 animate-pulse" />
            )}
          </div>
          <div>
            <h3 className="text-sm font-black uppercase tracking-[0.2em] text-emerald-400">Gemma 4 Edge</h3>
            <p className="text-[10px] text-slate-500 font-medium">Predictive Intelligence Layer</p>
          </div>
        </div>
        
        {expanded && (
          <button 
            onClick={(e) => { e.stopPropagation(); setExpanded(false); }}
            className="p-1.5 hover:bg-slate-800 rounded-lg text-slate-500 transition-colors"
          >
            <ChevronRight className="w-4 h-4 rotate-90" />
          </button>
        )}
      </div>

      {expanded ? (
        <div className="space-y-4 animate-in fade-in slide-in-from-top-2 duration-500">
          <div className="bg-slate-950/50 rounded-xl p-4 border border-slate-800/50">
            {loading ? (
              <div className="space-y-2 py-2">
                <div className="h-4 bg-slate-800 rounded animate-pulse w-3/4" />
                <div className="h-4 bg-slate-800 rounded animate-pulse w-1/2" />
                <div className="h-4 bg-slate-800 rounded animate-pulse w-5/6" />
              </div>
            ) : (
              <div className="text-sm leading-relaxed text-slate-300 whitespace-pre-line font-medium">
                {insight || "กำลังรอข้อมูลวิเคราะห์..."}
              </div>
            )}
          </div>

          <div className="flex items-center justify-between pt-2">
            <div className="flex gap-2">
              <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 text-[10px] font-bold border border-emerald-500/20">
                Decision Explainer
              </span>
              {type === 'incident' && (
                <span className="px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-400 text-[10px] font-bold border border-rose-500/20 flex items-center gap-1">
                  <ShieldAlert className="w-2.5 h-2.5" /> Action Plan
                </span>
              )}
            </div>
            <button className="text-[10px] font-bold text-slate-500 hover:text-emerald-400 flex items-center gap-1 transition-colors uppercase tracking-widest">
              See SOP Ref <ArrowRight className="w-3 h-3" />
            </button>
          </div>
        </div>
      ) : (
        <div className="flex items-center justify-between text-xs text-slate-400 font-medium italic">
          <span>Click to expand Gemma Insights...</span>
          <Info className="w-4 h-4 opacity-50" />
        </div>
      )}
    </div>
  );
};
