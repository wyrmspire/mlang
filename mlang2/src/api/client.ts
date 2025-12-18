import { VizDecision, VizTrade, RunManifest, AgentResponse, ChatMessage } from '../types/viz';

// API base URL - uses environment variable or defaults to localhost
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Flag to track if backend is available
let backendAvailable: boolean | null = null;

// Check backend availability
async function checkBackend(): Promise<boolean> {
    if (backendAvailable !== null) return backendAvailable;
    try {
        const response = await fetch(`${API_BASE}/health`, { method: 'GET' });
        backendAvailable = response.ok;
    } catch {
        backendAvailable = false;
    }
    return backendAvailable;
}

// ============================================================================
// MOCK DATA (fallback when backend unavailable)
// ============================================================================
const generateMockCandles = (count: number, startPrice: number): number[][] => {
    const candles: number[][] = [];
    let price = startPrice;
    for (let i = 0; i < count; i++) {
        const change = (Math.random() - 0.5) * 2;
        const open = price;
        const close = open + change;
        const high = Math.max(open, close) + Math.random();
        const low = Math.min(open, close) - Math.random();
        const volume = Math.floor(Math.random() * 1000 + 500);
        candles.push([open, high, low, close, volume]);
        price = close;
    }
    return candles;
};

const createMockData = () => {
    const decisions: VizDecision[] = [];
    const trades: VizTrade[] = [];
    let basePrice = 4500;

    for (let i = 0; i < 20; i++) {
        const isTrade = i % 5 === 0;
        const candles = generateMockCandles(100, basePrice);
        const currentPrice = candles[candles.length - 1][3];
        basePrice = currentPrice;

        const oco = isTrade ? {
            entry_price: currentPrice,
            stop_price: currentPrice - 10,
            tp_price: currentPrice + 20,
            entry_type: 'LIMIT',
            direction: 'LONG',
            reference_type: 'Price',
            reference_value: currentPrice,
            atr_at_creation: 2.5,
            max_bars: 200,
            stop_atr: 1,
            tp_multiple: 2
        } : undefined;

        decisions.push({
            decision_id: `dec_${i}`,
            timestamp: new Date(Date.now() - (20 - i) * 60000 * 15).toISOString(),
            bar_idx: 1000 + i * 15,
            index: i,
            scanner_id: isTrade ? 'opening_range' : 'wait',
            scanner_context: { rsi: 50 + Math.random() * 20 },
            action: isTrade ? 'PLACE_ORDER' : 'NO_TRADE',
            skip_reason: isTrade ? '' : 'filter_block',
            current_price: currentPrice,
            atr: 2.5,
            cf_outcome: isTrade ? 'WIN' : '',
            cf_pnl_dollars: isTrade ? 100 : 0,
            window: {
                x_price_1m: candles,
                x_price_5m: [],
                x_price_15m: [],
                x_context: [],
                norm_method: 'zscore',
                norm_params: {}
            },
            oco
        });

        if (isTrade) {
            const isWin = Math.random() > 0.4;
            trades.push({
                trade_id: `trd_${trades.length}`,
                decision_id: `dec_${i}`,
                index: trades.length,
                direction: 'LONG',
                size: 1,
                entry_time: decisions[i].timestamp,
                entry_bar: decisions[i].bar_idx + 1,
                entry_price: currentPrice,
                exit_time: null,
                exit_bar: decisions[i].bar_idx + 15,
                exit_price: isWin ? currentPrice + 20 : currentPrice - 10,
                exit_reason: isWin ? 'TP' : 'SL',
                outcome: isWin ? 'WIN' : 'LOSS',
                pnl_points: isWin ? 20 : -10,
                pnl_dollars: isWin ? 100 : -50,
                r_multiple: isWin ? 2.0 : -1.0,
                bars_held: 14,
                mae: -2,
                mfe: 22,
                fills: []
            });
        }
    }
    return { decisions, trades };
};

const mockData = createMockData();

// ============================================================================
// API CLIENT
// ============================================================================
export const api = {
    getRuns: async (): Promise<string[]> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            console.warn('Backend unavailable, using mock data');
            return ['demo_run'];
        }
        try {
            const response = await fetch(`${API_BASE}/runs`);
            if (!response.ok) return ['demo_run'];
            return response.json();
        } catch {
            return ['demo_run'];
        }
    },

    getRun: async (runId: string): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            return { id: runId, status: 'MOCK', stats: { total_trades: mockData.trades.length } };
        }
        try {
            const response = await fetch(`${API_BASE}/runs/${runId}`);
            if (!response.ok) throw new Error();
            return response.json();
        } catch {
            return { id: runId, status: 'MOCK' };
        }
    },

    getManifest: async (runId: string): Promise<RunManifest> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            return { run_id: runId, start_time: new Date().toISOString(), config: {} };
        }
        const response = await fetch(`${API_BASE}/runs/${runId}/manifest`);
        return response.json();
    },

    getDecisions: async (runId: string): Promise<VizDecision[]> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            console.warn('Using mock decisions');
            return mockData.decisions;
        }
        try {
            const response = await fetch(`${API_BASE}/runs/${runId}/decisions`);
            if (!response.ok) return mockData.decisions;
            return response.json();
        } catch {
            return mockData.decisions;
        }
    },

    getTrades: async (runId: string): Promise<VizTrade[]> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            console.warn('Using mock trades');
            return mockData.trades;
        }
        try {
            const response = await fetch(`${API_BASE}/runs/${runId}/trades`);
            if (!response.ok) return mockData.trades;
            return response.json();
        } catch {
            return mockData.trades;
        }
    },

    postAgent: async (
        messages: ChatMessage[],
        context: { runId: string, currentIndex: number, currentMode: 'DECISION' | 'TRADE' }
    ): Promise<AgentResponse> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            // Mock agent - simple responses
            const lastMsg = messages[messages.length - 1]?.content.toLowerCase() || '';
            if (lastMsg.includes('next')) {
                return { reply: 'Moving to next item.', ui_action: { type: 'SET_INDEX', payload: context.currentIndex + 1 } };
            }
            if (lastMsg.includes('prev')) {
                return { reply: 'Moving to previous item.', ui_action: { type: 'SET_INDEX', payload: Math.max(0, context.currentIndex - 1) } };
            }
            return { reply: 'Backend not connected. Start with: ./start.sh' };
        }
        try {
            const response = await fetch(`${API_BASE}/agent/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages, context }),
            });
            if (!response.ok) return { reply: 'Error contacting agent server.' };
            return response.json();
        } catch {
            return { reply: 'Error contacting agent server.' };
        }
    }
};