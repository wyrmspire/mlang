import React, { useState, useEffect, useRef } from 'react';
import { yfinanceApi, Trade } from '../api/yfinance';
import { Candle } from '../api/client';
import { YFinanceChart } from './YFinanceChart';
import { Play, Pause, RotateCcw } from 'lucide-react';

export const YFinanceMode: React.FC = () => {
    // Data state
    const [allData, setAllData] = useState<Candle[]>([]);
    const [dates, setDates] = useState<string[]>([]);
    const [selectedDate, setSelectedDate] = useState<string>('');
    const [symbol, setSymbol] = useState<string>('ES=F');
    
    // Playback state
    const [currentIndex, setCurrentIndex] = useState<number>(0);
    const [isPlaying, setIsPlaying] = useState<boolean>(false);
    const [playbackSpeed, setPlaybackSpeed] = useState<number>(100); // ms per candle
    const [visibleData, setVisibleData] = useState<Candle[]>([]);
    
    // Model & Trading state
    const [availableModels, setAvailableModels] = useState<string[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>('rejection_cnn_v1');
    const [riskAmount, setRiskAmount] = useState<number>(300);
    const [trades, setTrades] = useState<Trade[]>([]);
    const [totalPnL, setTotalPnL] = useState<number>(0);
    
    // UI state
    const [isLoading, setIsLoading] = useState<boolean>(false);
    
    // Refs
    const playbackIntervalRef = useRef<number | null>(null);

    // Load models on mount
    useEffect(() => {
        loadModels();
    }, []);

    const loadModels = async () => {
        try {
            const models = await yfinanceApi.getAvailableModels();
            setAvailableModels(models);
            if (models.length > 0 && !models.includes(selectedModel)) {
                setSelectedModel(models[0]);
            }
        } catch (e) {
            console.error('Error loading models:', e);
        }
    };

    const loadData = async () => {
        setIsLoading(true);
        try {
            const result = await yfinanceApi.fetchData(symbol, 14);
            if (result.success) {
                setAllData(result.data);
                setDates(result.dates);
                if (result.dates.length > 0 && !selectedDate) {
                    setSelectedDate(result.dates[0]);
                }
                // Reset playback
                setCurrentIndex(0);
                setVisibleData([]);
                setTrades([]);
                setTotalPnL(0);
            } else {
                alert(result.message || 'Failed to load data');
            }
        } catch (e) {
            console.error('Error loading data:', e);
            alert('Failed to load data from YFinance');
        } finally {
            setIsLoading(false);
        }
    };

    // Filter data by date
    const dateData = React.useMemo(() => {
        if (!selectedDate || allData.length === 0) return [];
        return allData.filter(c => {
            const date = new Date(c.time * 1000).toISOString().split('T')[0];
            return date === selectedDate;
        });
    }, [allData, selectedDate]);

    // Playback control
    useEffect(() => {
        if (isPlaying && dateData.length > 0 && currentIndex < dateData.length) {
            playbackIntervalRef.current = window.setInterval(() => {
                setCurrentIndex(prev => {
                    const next = prev + 1;
                    if (next >= dateData.length) {
                        setIsPlaying(false);
                        return prev;
                    }
                    return next;
                });
            }, playbackSpeed);
        } else {
            if (playbackIntervalRef.current !== null) {
                clearInterval(playbackIntervalRef.current);
                playbackIntervalRef.current = null;
            }
        }

        return () => {
            if (playbackIntervalRef.current !== null) {
                clearInterval(playbackIntervalRef.current);
            }
        };
    }, [isPlaying, currentIndex, dateData.length, playbackSpeed]);

    // Update visible data as playback progresses
    useEffect(() => {
        if (dateData.length > 0 && currentIndex >= 0) {
            setVisibleData(dateData.slice(0, currentIndex + 1));
        }
    }, [currentIndex, dateData]);

    // Check for trade signals as new candles appear
    useEffect(() => {
        // Need at least 20 candles for analysis (matches model WINDOW_SIZE)
        const MIN_CANDLES_FOR_ANALYSIS = 20;
        
        if (!isPlaying || currentIndex < MIN_CANDLES_FOR_ANALYSIS) return;

        const checkForSignal = async () => {
            try {
                const signal = await yfinanceApi.analyzeCandle(
                    currentIndex,
                    selectedModel,
                    symbol,
                    selectedDate
                );

                if (signal) {
                    // Create new trade
                    const newTrade: Trade = {
                        id: `trade_${Date.now()}_${Math.random()}`,
                        direction: signal.direction,
                        entry_price: signal.entry_price,
                        entry_time: signal.entry_time,
                        sl_price: signal.sl_price,
                        tp_price: signal.tp_price,
                        status: 'open',
                        risk_amount: riskAmount
                    };
                    setTrades(prev => [...prev, newTrade]);
                }
            } catch (e) {
                console.error('Error checking for signal:', e);
            }
        };

        checkForSignal();
    }, [currentIndex, isPlaying]);

    // Update open trades with current price
    useEffect(() => {
        if (visibleData.length === 0) return;

        const currentCandle = visibleData[visibleData.length - 1];
        const currentTime = currentCandle.time;

        setTrades(prevTrades => {
            let pnlChange = 0;
            const updatedTrades = prevTrades.map(trade => {
                if (trade.status !== 'open') return trade;

                // Check if SL or TP hit
                const isLong = trade.direction === 'LONG';
                
                let hitSL = false;
                let hitTP = false;

                if (isLong) {
                    hitSL = currentCandle.low <= trade.sl_price;
                    hitTP = currentCandle.high >= trade.tp_price;
                } else {
                    hitSL = currentCandle.high >= trade.sl_price;
                    hitTP = currentCandle.low <= trade.tp_price;
                }

                if (hitSL) {
                    // Stop loss hit - lose risk amount
                    const pnl = -trade.risk_amount;
                    pnlChange += pnl;
                    return {
                        ...trade,
                        exit_price: trade.sl_price,
                        exit_time: currentTime,
                        pnl: pnl,
                        status: 'closed' as const
                    };
                } else if (hitTP) {
                    // Take profit hit - win risk amount (1:1 R:R)
                    const pnl = trade.risk_amount;
                    pnlChange += pnl;
                    return {
                        ...trade,
                        exit_price: trade.tp_price,
                        exit_time: currentTime,
                        pnl: pnl,
                        status: 'closed' as const
                    };
                }

                return trade;
            });

            if (pnlChange !== 0) {
                setTotalPnL(prev => prev + pnlChange);
            }

            return updatedTrades;
        });
    }, [visibleData]);

    const handlePlayPause = () => {
        if (dateData.length === 0) {
            alert('Please load data first');
            return;
        }
        setIsPlaying(!isPlaying);
    };

    const handleReset = () => {
        setIsPlaying(false);
        setCurrentIndex(0);
        setVisibleData([]);
        setTrades([]);
        setTotalPnL(0);
    };

    const openTrades = trades.filter(t => t.status === 'open');
    const closedTrades = trades.filter(t => t.status === 'closed');
    const winningTrades = closedTrades.filter(t => (t.pnl || 0) > 0);
    const winRate = closedTrades.length > 0 ? (winningTrades.length / closedTrades.length * 100) : 0;

    return (
        <div style={{ display: 'flex', height: '100vh', background: '#1E1E1E', color: '#EEE' }}>
            {/* Sidebar */}
            <div style={{ width: '280px', background: '#252526', padding: '15px', borderRight: '1px solid #333', overflowY: 'auto' }}>
                <h2 style={{ margin: '0 0 20px 0', fontSize: '18px' }}>YFinance Playback</h2>

                {/* Symbol & Load */}
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>Symbol</label>
                    <input
                        type="text"
                        value={symbol}
                        onChange={e => setSymbol(e.target.value)}
                        style={{ width: '100%', padding: '8px', background: '#333', color: '#EEE', border: '1px solid #555', borderRadius: '4px' }}
                    />
                </div>

                <button
                    onClick={loadData}
                    disabled={isLoading}
                    style={{
                        width: '100%',
                        padding: '10px',
                        background: isLoading ? '#555' : '#0E639C',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: isLoading ? 'wait' : 'pointer',
                        marginBottom: '15px'
                    }}
                >
                    {isLoading ? 'Loading...' : 'Load Data'}
                </button>

                {dates.length > 0 && (
                    <div style={{ marginBottom: '15px' }}>
                        <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>Date</label>
                        <select
                            value={selectedDate}
                            onChange={e => {
                                setSelectedDate(e.target.value);
                                handleReset();
                            }}
                            style={{ width: '100%', padding: '8px', background: '#333', color: '#EEE', border: '1px solid #555', borderRadius: '4px' }}
                        >
                            {dates.map(d => <option key={d} value={d}>{d}</option>)}
                        </select>
                    </div>
                )}

                <hr style={{ borderColor: '#444', margin: '15px 0' }} />

                {/* Model Selection */}
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>Model</label>
                    <select
                        value={selectedModel}
                        onChange={e => setSelectedModel(e.target.value)}
                        style={{ width: '100%', padding: '8px', background: '#333', color: '#EEE', border: '1px solid #555', borderRadius: '4px' }}
                    >
                        {availableModels.map(m => <option key={m} value={m}>{m}</option>)}
                    </select>
                </div>

                {/* Risk Amount */}
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>Risk per Trade ($)</label>
                    <input
                        type="number"
                        value={riskAmount}
                        onChange={e => setRiskAmount(Number(e.target.value))}
                        min={50}
                        max={5000}
                        step={50}
                        style={{ width: '100%', padding: '8px', background: '#333', color: '#EEE', border: '1px solid #555', borderRadius: '4px' }}
                    />
                </div>

                {/* Playback Speed */}
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>Speed (ms/candle)</label>
                    <input
                        type="range"
                        value={playbackSpeed}
                        onChange={e => setPlaybackSpeed(Number(e.target.value))}
                        min={10}
                        max={1000}
                        step={10}
                        style={{ width: '100%' }}
                    />
                    <div style={{ fontSize: '11px', color: '#999', textAlign: 'center' }}>{playbackSpeed}ms</div>
                </div>

                <hr style={{ borderColor: '#444', margin: '15px 0' }} />

                {/* Statistics */}
                <div style={{ marginBottom: '15px' }}>
                    <h3 style={{ margin: '0 0 10px 0', fontSize: '14px' }}>Statistics</h3>
                    <div style={{ fontSize: '12px', lineHeight: '1.6' }}>
                        <div>Total P&L: <span style={{ color: totalPnL >= 0 ? '#4caf50' : '#ef5350', fontWeight: 'bold' }}>${totalPnL.toFixed(2)}</span></div>
                        <div>Open Trades: {openTrades.length}</div>
                        <div>Closed Trades: {closedTrades.length}</div>
                        <div>Win Rate: {winRate.toFixed(1)}%</div>
                        <div>Progress: {currentIndex} / {dateData.length}</div>
                    </div>
                </div>

                {/* Open Trades */}
                {openTrades.length > 0 && (
                    <div style={{ marginBottom: '15px' }}>
                        <h3 style={{ margin: '0 0 10px 0', fontSize: '14px' }}>Open Positions</h3>
                        {openTrades.map(trade => (
                            <div key={trade.id} style={{ background: '#333', padding: '8px', marginBottom: '5px', borderRadius: '4px', fontSize: '11px' }}>
                                <div style={{ fontWeight: 'bold' }}>{trade.direction}</div>
                                <div>Entry: {trade.entry_price.toFixed(2)}</div>
                                <div>SL: {trade.sl_price.toFixed(2)} | TP: {trade.tp_price.toFixed(2)}</div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Main Content */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '10px' }}>
                {/* Controls */}
                <div style={{ display: 'flex', gap: '10px', marginBottom: '10px', alignItems: 'center', background: '#252526', padding: '10px', borderRadius: '4px' }}>
                    <button
                        onClick={handlePlayPause}
                        disabled={dateData.length === 0}
                        style={{
                            padding: '10px 20px',
                            background: '#0E639C',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: dateData.length === 0 ? 'not-allowed' : 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '5px'
                        }}
                    >
                        {isPlaying ? <><Pause size={16} /> Pause</> : <><Play size={16} /> Play</>}
                    </button>

                    <button
                        onClick={handleReset}
                        style={{
                            padding: '10px 20px',
                            background: '#333',
                            color: '#EEE',
                            border: '1px solid #555',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '5px'
                        }}
                    >
                        <RotateCcw size={16} /> Reset
                    </button>

                    <div style={{ flex: 1 }} />

                    <div style={{ fontSize: '14px', color: '#999' }}>
                        {selectedDate && `${selectedDate} | ${currentIndex} / ${dateData.length} candles`}
                    </div>
                </div>

                {/* Chart */}
                <div style={{ flex: 1, border: '1px solid #444', borderRadius: '4px', overflow: 'hidden' }}>
                    <YFinanceChart
                        data={visibleData}
                        trades={trades}
                        currentPrice={visibleData.length > 0 ? visibleData[visibleData.length - 1].close : 0}
                    />
                </div>
            </div>
        </div>
    );
};
