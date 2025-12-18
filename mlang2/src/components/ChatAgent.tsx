import React, { useState, useRef, useEffect } from 'react';
import { api } from '../api/client';
import { ChatMessage, UIAction } from '../types/viz';

interface ChatAgentProps {
  runId: string;
  currentIndex: number;
  currentMode: 'DECISION' | 'TRADE';
  onAction: (action: UIAction) => void;
}

export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, currentMode, onAction }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'assistant', content: 'Hello. I am the Trade Viz Agent. How can I assist with the analysis?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !runId) return;

    const userMsg: ChatMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const response = await api.postAgent([...messages, userMsg], { runId, currentIndex, currentMode });
      
      setMessages(prev => [...prev, { role: 'assistant', content: response.reply }]);
      
      if (response.ui_action) {
        onAction(response.ui_action);
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: "Error contacting agent." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-800 border-t border-slate-700">
      <div className="px-3 py-2 bg-slate-900 border-b border-slate-700">
        <h3 className="text-xs font-bold text-blue-400 uppercase tracking-wider">Agent Chat</h3>
      </div>
      
      <div className="flex-1 overflow-y-auto p-3 space-y-3" ref={scrollRef}>
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${
              m.role === 'user' 
                ? 'bg-blue-600 text-white' 
                : 'bg-slate-700 text-slate-200'
            }`}>
              {m.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="text-xs text-slate-500 animate-pulse ml-2">Agent is thinking...</div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="p-2 border-t border-slate-700 flex gap-2">
        <input 
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask about this trade..."
          className="flex-1 bg-slate-900 border border-slate-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
        />
        <button 
          type="submit" 
          disabled={loading || !runId}
          className="bg-blue-600 hover:bg-blue-500 text-white rounded px-4 py-2 text-sm font-bold disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  );
};