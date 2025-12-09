import axios from 'axios';

const API_URL = 'http://localhost:8000';

export interface Candle {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    is_synthetic?: boolean;
    synthetic_day?: number;
}

export const api = {
    getDates: async () => {
        const res = await axios.get<string[]>(`${API_URL}/api/dates`);
        return res.data;
    },

    getCandles: async (date: string, timeframe: string = '1m', sessionType: string = 'all') => {
        const res = await axios.get<{ data: Candle[] }>(`${API_URL}/api/candles`, {
            params: { date, timeframe, session_type: sessionType }
        });
        return res.data.data;
    },

    getCandlesRange: async (startDate: string, endDate: string, timeframe: string, sessionType: string = 'all') => {
        const res = await axios.get<{ data: Candle[] }>(`${API_URL}/api/candles-range`, {
            params: { start_date: startDate, end_date: endDate, timeframe, session_type: sessionType }
        });
        return res.data.data;
    },
};
