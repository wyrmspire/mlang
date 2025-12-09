import axios from 'axios';
import { Candle } from './client';

const API_URL = 'http://localhost:8000';

export interface TradeSignal {
    type?: string; // 'OCO_LIMIT' | 'MARKET'
    direction: string;
    entry_price: number;
    sl_price: number; // For direct market orders
    tp_price: number; // For direct market orders
    confidence: number;
    entry_time: number;
    atr?: number;
    setup_type?: string;

    // OCO Fields
    sell_limit?: number;
    buy_limit?: number;
    sl_dist?: number;
    limit_dist?: number;
    current_price?: number;
}

export interface YFinanceData {
    success: boolean;
    data: Candle[];
    dates: string[];
    symbol: string;
    message?: string;
}

export interface Trade {
    id: string;
    direction: string;
    entry_price: number;
    entry_time: number;
    sl_price: number;
    tp_price: number;
    exit_price?: number;
    exit_time?: number;
    pnl?: number;
    status: 'open' | 'closed' | 'pending';
    risk_amount: number;
}

export const yfinanceApi = {
    fetchData: async (symbol: string = 'ES=F', daysBack: number = 7, interval: string = '1m', useMock: boolean = false): Promise<YFinanceData> => {
        const endpoint = useMock ? '/api/yfinance/candles/mock' : '/api/yfinance/candles';
        const params = useMock 
            ? { bars: daysBack * (interval === '1m' ? 390 : 78), timeframe: interval }  // ~390 1m bars per day, ~78 5m bars
            : { symbol, days: daysBack, timeframe: interval };
            
        const res = await axios.get<YFinanceData>(`${API_URL}${endpoint}`, { params });
        return res.data;
    },

    getAvailableModels: async (): Promise<string[]> => {
        const res = await axios.get<{ models: string[] }>(`${API_URL}/api/yfinance/models`);
        return res.data.models;
    },

    analyzeCandle: async (
        candleIndex: number,
        modelName: string,
        symbol: string,
        date: string
    ): Promise<{ signal: TradeSignal | null }> => {
        const res = await axios.post<{ signal: TradeSignal | null }>(
            `${API_URL}/api/yfinance/playback/analyze`,
            null,
            {
                params: {
                    candle_index: candleIndex,
                    model_name: modelName,
                    symbol,
                    date
                }
            }
        );
        // Return matching the structure expected by YFinanceMode (wrapper object)
        // Or change YFinanceMode. currently YFinanceMode expects `resp.signal`.
        // If I return `res.data`, it contains `signal`.
        return res.data;
    }
};
