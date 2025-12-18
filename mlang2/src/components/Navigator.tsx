import React from 'react';

interface NavigatorProps {
  mode: 'DECISION' | 'TRADE';
  setMode: (m: 'DECISION' | 'TRADE') => void;
  index: number;
  setIndex: (i: number) => void;
  maxIndex: number;
}

export const Navigator: React.FC<NavigatorProps> = ({ mode, setMode, index, setIndex, maxIndex }) => {
  return (
    <div className="p-4 border-b border-slate-700 flex flex-col gap-4">
      
      {/* Mode Toggle */}
      <div className="flex bg-slate-900 rounded p-1">
        <button 
          onClick={() => setMode('DECISION')}
          className={`flex-1 text-xs py-1.5 rounded font-medium transition-colors ${mode === 'DECISION' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
        >
          Decisions
        </button>
        <button 
          onClick={() => setMode('TRADE')}
          className={`flex-1 text-xs py-1.5 rounded font-medium transition-colors ${mode === 'TRADE' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
        >
          Trades
        </button>
      </div>

      {/* Navigation Controls */}
      <div>
        <div className="flex justify-between items-center mb-2">
          <span className="text-xs font-mono text-slate-400">
            IDX: <span className="text-white">{index}</span> / {maxIndex}
          </span>
        </div>
        
        <div className="flex gap-2 mb-2">
          <button 
            onClick={() => setIndex(Math.max(0, index - 1))}
            className="flex-1 bg-slate-700 hover:bg-slate-600 text-white py-1 rounded text-sm disabled:opacity-50"
            disabled={index <= 0}
          >
            Prev
          </button>
          <button 
            onClick={() => setIndex(Math.min(maxIndex, index + 1))}
            className="flex-1 bg-slate-700 hover:bg-slate-600 text-white py-1 rounded text-sm disabled:opacity-50"
            disabled={index >= maxIndex}
          >
            Next
          </button>
        </div>

        <input 
          type="range" 
          min="0" 
          max={maxIndex} 
          value={index} 
          onChange={(e) => setIndex(parseInt(e.target.value))}
          className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"
        />
      </div>

      <div className="relative">
          <input 
            type="text" 
            placeholder="Search ID..." 
            className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-blue-500"
          />
      </div>

    </div>
  );
};