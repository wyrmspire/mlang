import React from 'react';
import { VizDecision, VizTrade } from '../types/viz';

interface DetailsPanelProps {
  decision: VizDecision | null;
  trade: VizTrade | null;
}

export const DetailsPanel: React.FC<DetailsPanelProps> = ({ decision, trade }) => {
  if (!decision) return <div className="p-4 text-slate-500">No data selected</div>;

  return (
    <div className="h-full flex flex-col">
      <div className="bg-slate-800 border-b border-slate-700 px-4 py-2 flex items-center justify-between">
        <h3 className="font-bold text-sm text-slate-200">
          {decision.action} <span className="font-mono text-slate-500 text-xs ml-2">{decision.decision_id}</span>
        </h3>
        {trade && (
          <span className={`text-xs font-bold px-2 py-0.5 rounded ${trade.outcome === 'WIN' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
            {trade.outcome} (${trade.pnl_dollars})
          </span>
        )}
      </div>

      <div className="flex-1 overflow-auto p-4 grid grid-cols-2 gap-4">

        {/* Left: Stats */}
        <div className="space-y-4">
          <section>
            <h4 className="text-xs font-bold text-slate-500 uppercase mb-2">Decision Specs</h4>
            <div className="grid grid-cols-2 gap-y-1 text-xs">
              <span className="text-slate-400">Scanner:</span>
              <span className="text-right font-mono">{decision.scanner_id || 'unknown'}</span>
              <span className="text-slate-400">Price:</span>
              <span className="text-right font-mono">{decision.current_price?.toFixed?.(2) ?? decision.current_price ?? '-'}</span>
              <span className="text-slate-400">ATR:</span>
              <span className="text-right font-mono">{decision.atr?.toFixed?.(2) ?? decision.atr ?? '-'}</span>
              <span className="text-slate-400">Skip Reason:</span>
              <span className="text-right font-mono text-yellow-500">{decision.skip_reason || '-'}</span>
            </div>
          </section>

          {trade && (
            <section>
              <h4 className="text-xs font-bold text-slate-500 uppercase mb-2">Trade Performance</h4>
              <div className="grid grid-cols-2 gap-y-1 text-xs">
                <span className="text-slate-400">Entry:</span>
                <span className="text-right font-mono">{trade.entry_price?.toFixed?.(2) ?? trade.entry_price ?? '-'}</span>
                <span className="text-slate-400">Exit:</span>
                <span className="text-right font-mono">{trade.exit_price?.toFixed?.(2) ?? trade.exit_price ?? '-'}</span>
                <span className="text-slate-400">R-Multiple:</span>
                <span className="text-right font-mono">{trade.r_multiple?.toFixed?.(2) ?? '-'}R</span>
                <span className="text-slate-400">MAE / MFE:</span>
                <span className="text-right font-mono">{trade.mae?.toFixed?.(2) ?? '-'} / {trade.mfe?.toFixed?.(2) ?? '-'}</span>
              </div>
            </section>
          )}
        </div>

        {/* Right: Context JSON */}
        <div className="bg-slate-900 rounded p-2 overflow-auto text-xs font-mono border border-slate-700">
          <div className="text-slate-500 mb-1">// Scanner Context</div>
          <pre className="text-blue-300 whitespace-pre-wrap">
            {JSON.stringify(decision.scanner_context, null, 2)}
          </pre>
          {decision.oco && (
            <>
              <div className="text-slate-500 mt-2 mb-1">// OCO Params</div>
              <pre className="text-green-300 whitespace-pre-wrap">
                {JSON.stringify(decision.oco, null, 2)}
              </pre>
            </>
          )}
        </div>

      </div>
    </div>
  );
};