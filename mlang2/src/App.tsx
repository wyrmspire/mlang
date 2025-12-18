import React, { useState, useEffect, useMemo } from 'react';
import { api } from './api/client';
import { VizDecision, VizTrade, UIAction } from './types/viz';
import { RunPicker } from './components/RunPicker';
import { Navigator } from './components/Navigator';
import { CandleChart } from './components/CandleChart';
import { DetailsPanel } from './components/DetailsPanel';
import { ChatAgent } from './components/ChatAgent';

const App: React.FC = () => {
  const [currentRun, setCurrentRun] = useState<string>('');
  const [mode, setMode] = useState<'DECISION' | 'TRADE'>('DECISION');
  const [index, setIndex] = useState<number>(0);
  const [showRawData, setShowRawData] = useState<boolean>(false);

  const [decisions, setDecisions] = useState<VizDecision[]>([]);
  const [trades, setTrades] = useState<VizTrade[]>([]);

  // Data Loading
  useEffect(() => {
    if (currentRun) {
      Promise.all([
        api.getDecisions(currentRun),
        api.getTrades(currentRun)
      ]).then(([d, t]) => {
        setDecisions(d);
        setTrades(t);
        setIndex(0); // Reset index on run change
      });
    }
  }, [currentRun]);

  // Derived State
  const activeDecision = useMemo(() => {
    if (mode === 'DECISION') {
      return decisions.find(d => d.index === index) || decisions[index] || null;
    } else {
      const trade = trades.find(t => t.index === index);
      return trade ? decisions.find(d => d.decision_id === trade.decision_id) || null : null;
    }
  }, [mode, index, decisions, trades]);

  const activeTrade = useMemo(() => {
    if (mode === 'TRADE') {
      return trades.find(t => t.index === index) || null;
    } else {
      // Try to find a trade linked to this decision
      return activeDecision ? trades.find(t => t.decision_id === activeDecision.decision_id) || null : null;
    }
  }, [mode, index, trades, activeDecision]);

  // Agent Action Handler
  const handleAgentAction = async (action: UIAction) => {
    switch (action.type) {
      case 'SET_INDEX':
        setIndex(action.payload);
        break;
      case 'SET_MODE':
        setMode(action.payload);
        setIndex(0);
        break;
      case 'LOAD_RUN':
        setCurrentRun(action.payload);
        break;
      case 'RUN_STRATEGY':
        // Call backend to run strategy
        try {
          const response = await fetch('http://localhost:8000/agent/run-strategy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(action.payload)
          });
          const result = await response.json();
          if (result.success && result.run_id) {
            // Auto-load the new run
            setCurrentRun(result.run_id);
          }
        } catch (e) {
          console.error('Failed to run strategy:', e);
        }
        break;
      default:
        console.warn('Unknown action:', action);
    }
  };

  const maxIndex = mode === 'DECISION' ? decisions.length - 1 : trades.length - 1;

  // Always show main UI - no blocking load screen
  return (
    <div className="flex h-screen w-full bg-slate-900 overflow-hidden">

      {/* LEFT SIDEBAR */}
      <div className="w-80 flex flex-col border-r border-slate-700 bg-slate-800">
        <div className="h-16 flex items-center px-4 border-b border-slate-700">
          <h1 className="font-bold text-white text-lg">Trade Viz Agent</h1>
        </div>

        <RunPicker onSelect={setCurrentRun} />

        <Navigator
          mode={mode}
          setMode={setMode}
          index={index}
          setIndex={setIndex}
          maxIndex={Math.max(0, maxIndex)}
        />

        <div className="flex-1 overflow-auto p-2">
          {!currentRun ? (
            <div className="p-4 text-sm text-slate-400 text-center">
              <p>Select a run above to load data.</p>
              <p className="mt-2 text-xs text-slate-500">Or use the chat to navigate.</p>
            </div>
          ) : (
            <div className="p-4 text-xs text-slate-500 text-center">
              Loaded {decisions.length} decisions, {trades.length} trades.
            </div>
          )}

          {/* Raw Data Toggle */}
          <div className="mt-2 px-2">
            <button
              onClick={() => setShowRawData(!showRawData)}
              className="w-full text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 py-2 px-3 rounded"
            >
              {showRawData ? 'Hide Raw Data' : 'Show Raw Data'}
            </button>
          </div>
        </div>

        {/* CHAT AGENT (Bottom Left) */}
        <div className="h-1/3 min-h-[300px]">
          <ChatAgent
            runId={currentRun || 'none'}
            currentIndex={index}
            currentMode={mode}
            onAction={handleAgentAction}
          />
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div className="flex-1 flex flex-col">

        {/* CHART AREA */}
        <div className="flex-1 relative bg-slate-900">
          {activeDecision && (activeDecision.window?.x_price_1m || (activeDecision as any).x_price_1m) ? (
            <CandleChart decision={activeDecision} trade={activeTrade} />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center text-slate-500">
              <div className="text-center">
                <p className="text-lg mb-2">No chart data available</p>
                <p className="text-sm text-slate-600">Select a run and navigate to a decision</p>
              </div>
            </div>
          )}

          {/* Floating Info Overlay */}
          {activeDecision && (
            <div className="absolute top-4 left-4 bg-slate-800/80 backdrop-blur px-3 py-2 rounded border border-slate-700 text-xs shadow-lg pointer-events-none">
              <div className="font-mono text-white">{activeDecision?.timestamp}</div>
              <div className="text-blue-400 font-bold">{activeDecision?.scanner_id || 'unknown'}</div>
              <div className="text-slate-400">Index: {activeDecision?.index}</div>
            </div>
          )}
        </div>

        {/* RAW DATA PANEL or DETAILS PANEL */}
        <div className="h-72 border-t border-slate-700 bg-slate-800 overflow-auto">
          {showRawData ? (
            <div className="p-4">
              <h3 className="text-xs font-bold text-slate-400 uppercase mb-2">Raw Decision Data</h3>
              <pre className="text-xs text-green-400 bg-slate-900 p-3 rounded overflow-auto max-h-56 font-mono">
                {activeDecision ? JSON.stringify(activeDecision, null, 2) : 'No data selected'}
              </pre>
            </div>
          ) : (
            <DetailsPanel decision={activeDecision} trade={activeTrade} />
          )}
        </div>

      </div>

    </div>
  );
};

export default App;