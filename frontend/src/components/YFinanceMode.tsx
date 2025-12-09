import React, { useState, useEffect, useRef } from 'react';
import { yfinanceApi, Trade } from '../api/yfinance';
import { Candle } from '../api/client';
import { YFinanceChart } from './YFinanceChart';
import { Play, Pause, RotateCcw, ArrowLeft } from 'lucide-react';

interface YFinanceModeProps {
    onBack: () => void;
}

export const YFinanceMode: React.FC<YFinanceModeProps> = ({ onBack }) => {
    // Data settings
    const [sourceInterval, setSourceInterval] = useState<'1m' | '5m'>('1m');
    const [loadDays, setLoadDays] = useState<number>(5);

    // Data state
    const [allData, setAllData] = useState<Candle[]>([]); // Source candles (1m or 5m)
    const [symbol, setSymbol] = useState<string>('MES=F');

    // Playback state
    const [currentIndex, setCurrentIndex] = useState<number>(0);
    const [isPlaying, setIsPlaying] = useState<boolean>(false);
    const [playbackSpeed, setPlaybackSpeed] = useState<number>(200);
    const [displayTimeframe, setDisplayTimeframe] = useState<number>(15); // Minutes

    // Chart Data (Aggregated)
    const [chartCandles, setChartCandles] = useState<Candle[]>([]);
    const [currentPrice, setCurrentPrice] = useState<number>(0);

    // Model & Trading state
    const [availableModels, setAvailableModels] = useState<string[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>('CNN_Predictive_5m');
    const [riskAmount] = useState<number>(300);
    const [trades, setTrades] = useState<Trade[]>([]);
    const [totalPnL, setTotalPnL] = useState<number>(0);
    const [unrealizedPnL, setUnrealizedPnL] = useState<number>(0);

    // UI state
    const [isLoading, setIsLoading] = useState<boolean>(false);

    // Refs
    const playbackIntervalRef = useRef<number | null>(null);

    // Initial Load
    useEffect(() => {
        loadModels();
        // Adjust defaults based on source interval
        if (sourceInterval === '1m') {
            setLoadDays(5);
            setDisplayTimeframe(15);
        } else {
            setLoadDays(30);
            setDisplayTimeframe(60);
        }
    }, [sourceInterval]);

    const loadModels = async () => {
        try {
            const models = await yfinanceApi.getAvailableModels();
            setAvailableModels(models);
            if (models.length > 0 && !models.includes(selectedModel)) {
                // Prefer predictive
                const pred = models.find(m => m.includes('Predictive'));
                setSelectedModel(pred || models[0]);
            }
        } catch (e) {
            console.error('Error loading models:', e);
        }
    };

    const loadData = async () => {
        setIsLoading(true);
        try {
            // Using loadDays and sourceInterval
            const result = await yfinanceApi.fetchData(symbol, loadDays, sourceInterval);

            if (result.success) {
                setAllData(result.data);
                // Reset everything
                resetPlayback(result.data);
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

    const resetPlayback = (data: Candle[] = allData) => {
        if (data.length === 0) return;

        // Start from beginning of loaded buffer? Or 200 bars in?
        // Let's start a bit in so we have history for the chart/model
        const startIdx = Math.min(200, data.length - 1);
        setCurrentIndex(startIdx);

        setIsPlaying(false);
        setChartCandles([]);
        setTrades([]);
        setTotalPnL(0);
        setUnrealizedPnL(0);
    };

    // Playback Loop
    useEffect(() => {
        if (isPlaying && allData.length > 0 && currentIndex < allData.length) {
            playbackIntervalRef.current = window.setInterval(() => {
                setCurrentIndex(prev => {
                    if (prev + 1 >= allData.length) {
                        setIsPlaying(false);
                        return prev;
                    }
                    return prev + 1;
                });
            }, playbackSpeed);
        } else {
            if (playbackIntervalRef.current !== null) {
                clearInterval(playbackIntervalRef.current);
                playbackIntervalRef.current = null;
            }
        }
        return () => {
            if (playbackIntervalRef.current !== null) clearInterval(playbackIntervalRef.current);
        };
    }, [isPlaying, currentIndex, allData.length, playbackSpeed]);

    // Update Chart & Logic on Update
    useEffect(() => {
        if (allData.length === 0 || currentIndex >= allData.length) return;

        const tick = allData[currentIndex];
        setCurrentPrice(tick.close);

        // 1. Accumulate Candle
        setChartCandles(prev => {
            if (prev.length === 0) {
                // Start with first tick aligned to timeframe boundary
                const tfSeconds = displayTimeframe * 60;
                const alignedTime = Math.floor(tick.time / tfSeconds) * tfSeconds;
                return [{
                    time: alignedTime,
                    open: tick.open,
                    high: tick.high,
                    low: tick.low,
                    close: tick.close,
                    volume: tick.volume
                }];
            }

            const lastCandle = prev[prev.length - 1];
            const tfSeconds = displayTimeframe * 60;
            const tickAlignedTime = Math.floor(tick.time / tfSeconds) * tfSeconds;

            // Check if tick belongs to current candle or starts new one
            if (tickAlignedTime === lastCandle.time) {
                // Update existing candle
                const updated = {
                    ...lastCandle,
                    high: Math.max(lastCandle.high, tick.high),
                    low: Math.min(lastCandle.low, tick.low),
                    close: tick.close,
                    volume: lastCandle.volume + tick.volume
                };
                return [...prev.slice(0, -1), updated];
            } else {
                // New candle starts
                return [...prev, {
                    time: tickAlignedTime,
                    open: tick.open,
                    high: tick.high,
                    low: tick.low,
                    close: tick.close,
                    volume: tick.volume
                }];
            }
        });

        // 2. Manage Trades
        updateTrades(tick);

        // 3. Check Signals - Only check when we complete a candle of the MODEL's input timeframe
        // For 5m mode: sourceInterval='1m', we check every 5 bars (5 minutes)
        // For 15m mode: sourceInterval='5m', we check every 3 bars (15 minutes)
        if (isPlaying && shouldCheckSignal(currentIndex)) {
            checkSignal(currentIndex);
        }

    }, [currentIndex]); // Only on index change
    
    // Helper to determine if we should check for signals at this index
    const shouldCheckSignal = (idx: number): boolean => {
        if (idx < 200) return false; // Need sufficient history
        
        // Check based on model's required interval
        // Models expect 20 candles of their input timeframe
        const checkInterval = sourceInterval === '1m' ? 5 : 3; // 5m for 1m data, 15m for 5m data
        
        // Check every N bars where N = minutes in target TF / minutes in source TF
        return idx % checkInterval === 0;
    };

    // Trade Management
    const updateTrades = (tick: Candle) => {
        setTrades(prev => {
            let pnlRealized = 0;
            const nextTrades = prev.map(t => {
                // Open Positions
                if (t.status === 'open') {
                    let hitSL = false, hitTP = false;
                    // Check SL/TP
                    if (t.direction === 'LONG') {
                        if (tick.low <= t.sl_price) hitSL = true;
                        else if (tick.high >= t.tp_price) hitTP = true;
                    } else if (t.direction === 'SHORT') {
                        if (tick.high >= t.sl_price) hitSL = true;
                        else if (tick.low <= t.tp_price) hitTP = true;
                    }

                    if (hitSL) {
                        const loss = -Math.abs(t.risk_amount);
                        pnlRealized += loss;
                        return { ...t, status: 'closed', pnl: loss, exit_price: t.sl_price, exit_time: tick.time } as Trade;
                    }
                    if (hitTP) {
                        // Calculate Reward based on Distances
                        const risk = Math.abs(t.entry_price - t.sl_price);
                        const reward = Math.abs(t.entry_price - t.tp_price);
                        const rMultiple = reward / (risk || 1);
                        const win = t.risk_amount * rMultiple;

                        pnlRealized += win;
                        return { ...t, status: 'closed', pnl: win, exit_price: t.tp_price, exit_time: tick.time } as Trade;
                    }
                }

                return t;
            });

            // Handle Pending Orders - separate pass to handle OCO logic
            let ocoGroupFilled: Record<string, boolean> = {};
            const finalTrades = nextTrades.map(t => {
                if (t.status === 'pending') {
                    // Check if order is before entry time
                    if (tick.time < t.entry_time) return t;

                    // Extract OCO group from ID (format: "s_123" or "b_123")
                    const groupId = t.id.substring(2); // Remove "s_" or "b_" prefix
                    
                    // Check if other side of OCO was already filled
                    if (ocoGroupFilled[groupId]) {
                        // Cancel this pending order (OCO)
                        return { ...t, status: 'closed', pnl: 0, exit_time: tick.time } as Trade;
                    }

                    // Check for expiry (15 minutes = 900 seconds)
                    const elapsedTime = tick.time - t.entry_time;
                    if (elapsedTime > 900) {
                        // Expired
                        return { ...t, status: 'closed', pnl: 0, exit_time: tick.time } as Trade;
                    }

                    // Check if filled
                    let filled = false;
                    if (t.direction === 'SELL' && tick.high >= t.entry_price) filled = true;
                    else if (t.direction === 'BUY' && tick.low <= t.entry_price) filled = true;

                    if (filled) {
                        // Mark this OCO group as filled
                        ocoGroupFilled[groupId] = true;
                        // Convert to position: BUY -> LONG, SELL -> SHORT
                        const positionDirection = t.direction === 'BUY' ? 'LONG' : 'SHORT';
                        return { ...t, status: 'open', direction: positionDirection } as Trade;
                    }
                }

                return t;
            });

            if (pnlRealized !== 0) setTotalPnL(s => s + pnlRealized);

            // Unrealized
            let float = 0;
            finalTrades.forEach(t => {
                if (t.status === 'open') {
                    let dist = 0;
                    if (t.direction === 'LONG') {
                        dist = tick.close - t.entry_price;
                    } else if (t.direction === 'SHORT') {
                        dist = t.entry_price - tick.close;
                    }
                    const risk = Math.abs(t.entry_price - t.sl_price);
                    const size = t.risk_amount / (risk || 1);
                    float += dist * size;
                }
            });
            setUnrealizedPnL(float);

            return finalTrades;
        });
    };

    const checkSignal = async (idx: number) => {
        try {
            // current timestamp
            const dateStr = new Date(allData[idx].time * 1000).toISOString().split('T')[0];

            const resp = await yfinanceApi.analyzeCandle(idx, selectedModel, symbol, dateStr);
            if (resp && resp.signal) {
                const s = resp.signal;

                // OCO Logic
                if (s.type === 'OCO_LIMIT') {
                    // Two pending orders
                    const sell: Trade = {
                        id: `s_${idx}`,
                        direction: 'SELL',
                        entry_price: s.sell_limit || 0,
                        sl_price: (s.sell_limit || 0) + (s.sl_dist || 0),
                        tp_price: s.current_price || 0, // Target Reversion to Mean
                        entry_time: allData[idx].time,
                        status: 'pending',
                        risk_amount: riskAmount
                    };
                    const buy: Trade = {
                        id: `b_${idx}`,
                        direction: 'BUY',
                        entry_price: s.buy_limit || 0,
                        sl_price: (s.buy_limit || 0) - (s.sl_dist || 0),
                        tp_price: s.current_price || 0,
                        entry_time: allData[idx].time,
                        status: 'pending',
                        risk_amount: riskAmount
                    };

                    // Avoid duplicates?
                    // Simple check: don't add if we just added same ID
                    setTrades(prev => {
                        if (prev.some(p => p.id === `s_${idx}`)) return prev;
                        return [...prev, sell, buy];
                    });
                }
            }
        } catch (e) { console.error(e); }
    };

    const openTrades = trades.filter(t => t.status === 'open');
    const pendingTrades = trades.filter(t => t.status === 'pending');

    return (
        <div style={{ display: 'flex', height: '100vh', background: '#1E1E1E', color: '#EEE' }}>
            {/* Sidebar */}
            <div style={{ width: '300px', background: '#252526', padding: '15px', borderRight: '1px solid #333', overflowY: 'auto' }}>
                <button onClick={onBack} style={{ display: 'flex', alignItems: 'center', gap: '5px', padding: '5px 10px', marginBottom: '20px', background: 'transparent', color: '#aaa', border: 'none', cursor: 'pointer' }}>
                    <ArrowLeft size={16} /> Back
                </button>

                <h2 style={{ margin: '0 0 20px 0' }}>True Replay</h2>

                <div className="control-group" style={{ marginBottom: '20px' }}>
                    <label>Symbol</label>
                    <input value={symbol} onChange={e => setSymbol(e.target.value)} style={{ width: '100%', padding: '5px' }} />

                    <label style={{ marginTop: '10px', display: 'block' }}>Source Granularity</label>
                    <div style={{ display: 'flex', gap: '5px' }}>
                        <button
                            onClick={() => setSourceInterval('1m')}
                            style={{ flex: 1, padding: '5px', background: sourceInterval === '1m' ? '#007acc' : '#444', border: 'none', color: 'white' }}>
                            1 Minute
                        </button>
                        <button
                            onClick={() => setSourceInterval('5m')}
                            style={{ flex: 1, padding: '5px', background: sourceInterval === '5m' ? '#007acc' : '#444', border: 'none', color: 'white' }}>
                            5 Minute
                        </button>
                    </div>

                    <label style={{ marginTop: '10px', display: 'block' }}>Days to Load</label>
                    <input type="number" value={loadDays} onChange={e => setLoadDays(Number(e.target.value))} min={1} max={60} style={{ width: '100%', padding: '5px' }} />
                    <div style={{ fontSize: '10px', color: '#777' }}>Max: ~7d (1m), ~60d (5m)</div>

                    <button onClick={loadData} disabled={isLoading} style={{ width: '100%', marginTop: '5px', padding: '8px', background: '#007acc', color: 'white', border: 'none' }}>
                        {isLoading ? 'Loading...' : 'Load Data'}
                    </button>
                </div>

                <div className="control-group" style={{ marginBottom: '20px' }}>
                    <label>Model</label>
                    <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={{ width: '100%', padding: '5px' }}>
                        {availableModels.map(m => <option key={m}>{m}</option>)}
                    </select>
                </div>

                <div className="control-group" style={{ marginBottom: '20px' }}>
                    <label>Chart Timeframe</label>
                    <select value={displayTimeframe} onChange={e => {
                        setDisplayTimeframe(Number(e.target.value));
                        setChartCandles([]);
                    }} style={{ width: '100%', padding: '5px' }}>
                        <option value={5} disabled={sourceInterval === '5m'}>5 Minutes</option>
                        <option value={15}>15 Minutes</option>
                        <option value={60}>1 Hour</option>
                    </select>
                </div>

                {/* Stats */}
                <div style={{ background: '#333', padding: '10px', borderRadius: '4px' }}>
                    <div>Realized PnL: <strong style={{ color: totalPnL >= 0 ? '#4caf50' : '#f44336' }}>${totalPnL.toFixed(2)}</strong></div>
                    <div>Floating PnL: <strong style={{ color: unrealizedPnL >= 0 ? '#4caf50' : '#f44336' }}>${unrealizedPnL.toFixed(2)}</strong></div>
                    <div>Open: {openTrades.length} | Pending: {pendingTrades.length}</div>
                </div>
            </div>

            {/* Main Chart */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                <div style={{ padding: '10px', display: 'flex', gap: '10px', alignItems: 'center', background: '#333' }}>
                    <button onClick={() => setIsPlaying(!isPlaying)} style={{ display: 'flex', alignItems: 'center', gap: '5px', padding: '5px 15px', cursor: 'pointer' }}>
                        {isPlaying ? <Pause size={16} /> : <Play size={16} />} {isPlaying ? 'Pause' : 'Play'}
                    </button>
                    <button onClick={() => resetPlayback()} style={{ display: 'flex', alignItems: 'center', gap: '5px', padding: '5px 15px', cursor: 'pointer' }}>
                        <RotateCcw size={16} /> Reset
                    </button>
                    <input type="range" min="10" max="1000" step="10" value={playbackSpeed} onChange={e => setPlaybackSpeed(Number(e.target.value))} />
                    <span>{playbackSpeed}ms</span>
                    <span style={{ marginLeft: 'auto' }}>{allData[currentIndex]?.time ? new Date(allData[currentIndex].time * 1000).toLocaleString() : '--'}</span>
                </div>

                <div style={{ flex: 1 }}>
                    <YFinanceChart
                        data={chartCandles}
                        trades={trades}
                        currentPrice={currentPrice}
                    />
                </div>
            </div>
        </div>
    );
}
