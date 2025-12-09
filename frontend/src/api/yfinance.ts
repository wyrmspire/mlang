import axios from 'axios';
import { Candle } from './client';

const API_URL = 'http://localhost:8000';

export interface TradeSignal {
    direction: string;
    entry_price: number;
    sl_price: number;
    tp_price: number;
    confidence: number;
    entry_time: number;
    atr?: number;
    setup_type?: string;
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
    status: 'open' | 'closed';
    risk_amount: number;
}

export const yfinanceApi = {
    fetchData: async (symbol: string = 'ES=F', daysBack: number = 7): Promise<YFinanceData> => {
        const res = await axios.post<YFinanceData>(`${API_URL}/api/yfinance/fetch`, {
            symbol,
            days_back: daysBack
        });
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
    ): Promise<TradeSignal | null> => {
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
        return res.data.signal;
    }
};
