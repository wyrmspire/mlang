import React, { useEffect, useState } from 'react';
import { api } from '../api/client';

export const RunPicker: React.FC<{ onSelect: (id: string) => void }> = ({ onSelect }) => {
  const [runs, setRuns] = useState<string[]>([]);
  
  useEffect(() => {
    api.getRuns().then(setRuns);
  }, []);

  return (
    <div className="p-4 border-b border-slate-700 bg-slate-800">
      <label className="text-xs text-slate-400 font-bold uppercase block mb-1">Select Run</label>
      <select 
        className="w-full bg-slate-900 border border-slate-700 rounded text-sm p-2 text-slate-200 focus:outline-none focus:border-blue-500"
        onChange={(e) => onSelect(e.target.value)}
      >
        <option value="">-- Choose Run --</option>
        {runs.map(r => <option key={r} value={r}>{r}</option>)}
      </select>
    </div>
  );
};