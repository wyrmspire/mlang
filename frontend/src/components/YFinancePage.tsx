import React, { useState } from 'react';
import { ChartPanel } from './ChartPanel';
import { api, Candle } from '../api/client';

interface YFinancePageProps {
    onBack: () => void;
}

export const YFinancePage: React.FC<YFinancePageProps> = ({ onBack }) => {
    const [symbol, setSymbol] = useState<string>('MES=F');
    const [days, setDays] = useState<number>(5);
    const [timeframe, setTimeframe] = useState<string>('1m');
    const [candles, setCandles] = useState<Candle[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleFetch = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const data = await api.getYFinanceCandles(symbol, days, timeframe);
            setCandles(data);
        } catch (e: any) {
            setError(e.response?.data?.detail || e.message || 'Failed to fetch data');
            setCandles([]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div style={{ display: 'flex', height: '100vh', background: '#1E1E1E', color: '#EEE', flexDirection: 'column' }}>
            {/* Header Bar */}
            <div style={{ 
                padding: '15px', 
                background: '#2D2D2D', 
                borderBottom: '1px solid #444',
                display: 'flex',
                gap: '15px',
                alignItems: 'center',
                flexWrap: 'wrap'
            }}>
                <button
                    onClick={onBack}
                    style={{
                        padding: '8px 15px',
                        background: '#555',
                        color: '#FFF',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '14px'
                    }}
                >
                    ← Back to Generator
                </button>

                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                    <label style={{ fontSize: '14px' }}>
                        Symbol:
                        <input
                            type="text"
                            value={symbol}
                            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                            placeholder="MES=F"
                            style={{
                                marginLeft: '5px',
                                padding: '6px 10px',
                                background: '#333',
                                color: '#FFF',
                                border: '1px solid #555',
                                borderRadius: '4px',
                                width: '120px'
                            }}
                        />
                    </label>

                    <label style={{ fontSize: '14px' }}>
                        Days:
                        <input
                            type="number"
                            value={days}
                            onChange={(e) => setDays(Math.max(1, parseInt(e.target.value) || 1))}
                            min="1"
                            max="30"
                            style={{
                                marginLeft: '5px',
                                padding: '6px 10px',
                                background: '#333',
                                color: '#FFF',
                                border: '1px solid #555',
                                borderRadius: '4px',
                                width: '60px'
                            }}
                        />
                    </label>

                    <label style={{ fontSize: '14px' }}>
                        Timeframe:
                        <select
                            value={timeframe}
                            onChange={(e) => setTimeframe(e.target.value)}
                            style={{
                                marginLeft: '5px',
                                padding: '6px 10px',
                                background: '#333',
                                color: '#FFF',
                                border: '1px solid #555',
                                borderRadius: '4px'
                            }}
                        >
                            <option value="1m">1 Minute</option>
                            <option value="5m">5 Minutes</option>
                            <option value="15m">15 Minutes</option>
                            <option value="60m">1 Hour</option>
                        </select>
                    </label>

                    <button
                        onClick={handleFetch}
                        disabled={isLoading}
                        style={{
                            padding: '8px 15px',
                            background: isLoading ? '#666' : '#4CAF50',
                            color: '#FFF',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: isLoading ? 'default' : 'pointer',
                            fontSize: '14px',
                            fontWeight: 'bold'
                        }}
                    >
                        {isLoading ? 'Loading...' : 'Fetch Data'}
                    </button>
                </div>
            </div>

            {/* Error Message */}
            {error && (
                <div style={{
                    padding: '10px 15px',
                    background: '#C33',
                    color: '#FFF',
                    fontSize: '14px',
                    borderBottom: '1px solid #900'
                }}>
                    ❌ {error}
                </div>
            )}

            {/* Chart Panel */}
            <div style={{ flex: 1, padding: '10px', overflow: 'hidden' }}>
                <ChartPanel 
                    data={candles} 
                    title={candles.length > 0 ? `${symbol} - ${timeframe} (${candles.length} candles)` : 'No data loaded'} 
                />
            </div>

            {/* Footer Info */}
            {candles.length > 0 && (
                <div style={{
                    padding: '10px 15px',
                    background: '#2D2D2D',
                    borderTop: '1px solid #444',
                    fontSize: '12px',
                    color: '#AAA'
                }}>
                    Loaded {candles.length} candles • 
                    {' '}First: {new Date(candles[0].time * 1000).toLocaleString()} • 
                    {' '}Last: {new Date(candles[candles.length - 1].time * 1000).toLocaleString()}
                </div>
            )}
        </div>
    );
};
