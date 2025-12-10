import React, { useState, useEffect, useRef } from 'react';
import { yfinanceApi, Trade } from '../api/yfinance';
import { Candle } from '../api/client';
import { YFinanceChart } from './YFinanceChart';
import { Play, Pause, RotateCcw, ArrowLeft, HelpCircle, ChevronDown, ChevronRight } from 'lucide-react';

interface YFinanceModeProps {
    onBack: () => void;
}

export const YFinanceMode: React.FC<YFinanceModeProps> = ({ onBack }) => {
    // Data settings
    const [sourceInterval, setSourceInterval] = useState<'1m' | '5m'>('1m');
    const [loadDays, setLoadDays] = useState<number>(5);
    const [useMockData, setUseMockData] = useState<boolean>(true); // Default to mock for testing

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
    const riskAmount = 300; // Fixed risk per trade
    const [trades, setTrades] = useState<Trade[]>([]);
    const [totalPnL, setTotalPnL] = useState<number>(0);
    const [unrealizedPnL, setUnrealizedPnL] = useState<number>(0);

    // Analysis State
    const [lastAnalysis, setLastAnalysis] = useState<{ prob: number, atr: number, time: number } | null>(null);

    // Entry Control State
    const [modelThreshold, setModelThreshold] = useState<number>(0.15);
    const [limitFactor, setLimitFactor] = useState<number>(1.5);
    const [slFactor, setSlFactor] = useState<number>(1.0);
    const [entryMechanism, setEntryMechanism] = useState<'Predictive Limit' | 'Market Close'>('Predictive Limit');

    // UI state
    const [isLoading, setIsLoading] = useState<boolean>(false);
    
    // Collapsible sections
    const [dataExpanded, setDataExpanded] = useState<boolean>(true);
    const [modelExpanded, setModelExpanded] = useState<boolean>(true);
    const [entryExpanded, setEntryExpanded] = useState<boolean>(true);
    const [statsExpanded, setStatsExpanded] = useState<boolean>(true);

    // Tooltip state
    const [showHelp, setShowHelp] = useState<{ [key: string]: boolean }>({});

    // Constants
    const ORDER_EXPIRY_SECONDS = 15 * 60; // 15 minutes

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
            const result = await yfinanceApi.fetchData(symbol, loadDays, sourceInterval, useMockData);

            if (result.data && result.data.length > 0) {
                setAllData(result.data);
                // Reset everything
                resetPlayback(result.data);
            } else {
                alert(result.message || 'Failed to load data');
            }
        } catch (e) {
            console.error('Error loading data:', e);
            alert('Failed to load data. Try enabling Mock Data mode.');
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
                    // MATCHING TEST SCRIPT: TP Check FIRST (Optimistic)
                    if (t.direction === 'LONG') {
                        if (tick.high >= t.tp_price) hitTP = true;
                        else if (tick.low <= t.sl_price) hitSL = true;
                    } else if (t.direction === 'SHORT') {
                        if (tick.low <= t.tp_price) hitTP = true;
                        else if (tick.high >= t.sl_price) hitSL = true;
                    } else {
                        console.error(`Unknown position direction: ${t.direction}`);
                        return { ...t, status: 'closed', pnl: 0, exit_time: tick.time } as Trade;
                    }

                    if (hitTP) {
                        // Win
                        const risk = Math.abs(t.entry_price - t.sl_price);
                        const reward = Math.abs(t.entry_price - t.tp_price);
                        const rMultiple = reward / (risk || 1);
                        const win = t.risk_amount * rMultiple;

                        pnlRealized += win;
                        return { ...t, status: 'closed', pnl: win, exit_price: t.tp_price, exit_time: tick.time } as Trade;
                    }
                    if (hitSL) {
                        const loss = -Math.abs(t.risk_amount);
                        pnlRealized += loss;
                        return { ...t, status: 'closed', pnl: loss, exit_price: t.sl_price, exit_time: tick.time } as Trade;
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

                    // Check for expiry (15 minutes)
                    const elapsedTime = tick.time - t.entry_time;
                    if (elapsedTime > ORDER_EXPIRY_SECONDS) {
                        // Expired
                        return { ...t, status: 'closed', pnl: 0, exit_time: tick.time } as Trade;
                    }

                    // Check if filled
                    let filled = false;
                    let fillPrice = t.entry_price;

                    if (t.direction === 'SELL') {
                        if (tick.high >= t.entry_price) {
                            filled = true;
                            // Gap Check: If Open is HIGHER than limit, we get filled at Open (Better Price)
                            fillPrice = Math.max(tick.open, t.entry_price);
                        }
                    } else if (t.direction === 'BUY') {
                        if (tick.low <= t.entry_price) {
                            filled = true;
                            // Gap Check: If Open is LOWER than limit, we get filled at Open (Better Price)
                            fillPrice = Math.min(tick.open, t.entry_price);
                        }
                    }

                    if (filled) {
                        // Mark this OCO group as filled
                        ocoGroupFilled[groupId] = true;
                        // Convert to position: BUY -> LONG, SELL -> SHORT
                        const positionDirection = t.direction === 'BUY' ? 'LONG' : 'SHORT';
                        // Update entry_price to the actual fill price
                        return { ...t, status: 'open', direction: positionDirection, entry_price: fillPrice } as Trade;
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
            const dateStr = new Date(allData[idx].time * 1000).toISOString().split('T')[0];
            const resp = await yfinanceApi.analyzeCandle(idx, selectedModel, symbol, dateStr);

            if (resp && resp.signal) {
                const s = resp.signal;
                const prob = s.prob || 0;
                const atr = s.atr || 0;

                setLastAnalysis({ prob, atr, time: allData[idx].time });

                // Frontend Decision Logic (Phase 12)
                if (prob < modelThreshold) return; // Sensitivity Check

                const currentPrice = allData[idx].close;

                if (entryMechanism === 'Predictive Limit') {
                    // Create Limit Orders based on Factors
                    const limitDist = limitFactor * atr;
                    const slDist = slFactor * atr;

                    const sellLimit = currentPrice + limitDist;
                    const buyLimit = currentPrice - limitDist;

                    const sell: Trade = {
                        id: `s_${idx}`,
                        direction: 'SELL',
                        entry_price: sellLimit,
                        sl_price: sellLimit + slDist,
                        tp_price: currentPrice, // Target Mean
                        entry_time: allData[idx].time,
                        status: 'pending',
                        risk_amount: riskAmount
                    };
                    const buy: Trade = {
                        id: `b_${idx}`,
                        direction: 'BUY',
                        entry_price: buyLimit,
                        sl_price: buyLimit - slDist,
                        tp_price: currentPrice,
                        entry_time: allData[idx].time,
                        status: 'pending',
                        risk_amount: riskAmount
                    };

                    setTrades(prev => [...prev, sell, buy]);
                }
            } else {
                // No signal (Warmup or Error)
                setLastAnalysis({ prob: 0, atr: 0, time: allData[idx].time });
            }
        } catch (e) {
            console.error(e);
            setLastAnalysis({ prob: -1, atr: 0, time: allData[idx].time }); // Error state
        }
    };

    const openTrades = trades.filter(t => t.status === 'open');
    const pendingTrades = trades.filter(t => t.status === 'pending');
    const closedTrades = trades.filter(t => t.status === 'closed');
    
    // Helper to render section header with collapse toggle
    const SectionHeader: React.FC<{ 
        title: string; 
        expanded: boolean; 
        onToggle: () => void;
        helpKey?: string;
        helpText?: string;
    }> = ({ title, expanded, onToggle, helpKey, helpText }) => (
        <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between',
            marginBottom: '10px',
            paddingBottom: '5px',
            borderBottom: '1px solid #444'
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <button 
                    onClick={onToggle}
                    style={{ 
                        background: 'transparent', 
                        border: 'none', 
                        color: '#aaa', 
                        cursor: 'pointer',
                        padding: '0',
                        display: 'flex',
                        alignItems: 'center'
                    }}
                >
                    {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                </button>
                <h3 style={{ fontSize: '14px', fontWeight: 'bold', margin: 0 }}>{title}</h3>
                {helpKey && helpText && (
                    <button
                        onClick={() => setShowHelp(prev => ({ ...prev, [helpKey]: !prev[helpKey] }))}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            color: '#777',
                            cursor: 'pointer',
                            padding: '2px',
                            display: 'flex'
                        }}
                        title="Show help"
                    >
                        <HelpCircle size={14} />
                    </button>
                )}
            </div>
        </div>
    );
    
    // Helper to render help text
    const HelpText: React.FC<{ helpKey: string; text: string }> = ({ helpKey, text }) => (
        showHelp[helpKey] ? (
            <div style={{
                background: '#2a2a2a',
                padding: '8px',
                borderRadius: '4px',
                fontSize: '11px',
                color: '#aaa',
                marginBottom: '10px',
                border: '1px solid #444'
            }}>
                {text}
            </div>
        ) : null
    );

    return (
        <div style={{ display: 'flex', height: '100vh', background: '#1E1E1E', color: '#EEE' }}>
            {/* Sidebar */}
            <div style={{ width: '320px', background: '#252526', padding: '15px', borderRight: '1px solid #333', overflowY: 'auto' }}>
                <button onClick={onBack} style={{ display: 'flex', alignItems: 'center', gap: '5px', padding: '5px 10px', marginBottom: '20px', background: 'transparent', color: '#aaa', border: 'none', cursor: 'pointer' }}>
                    <ArrowLeft size={16} /> Back
                </button>

                <h2 style={{ margin: '0 0 20px 0', fontSize: '18px' }}>YFinance Playback</h2>
                
                {/* Data Source Section */}
                <div style={{ marginBottom: '20px' }}>
                    <SectionHeader 
                        title="Data Source" 
                        expanded={dataExpanded} 
                        onToggle={() => setDataExpanded(!dataExpanded)}
                        helpKey="dataSource"
                        helpText="Load real market data from YFinance or use mock data for testing. Mock data is recommended for initial testing."
                    />
                    <HelpText 
                        helpKey="dataSource" 
                        text="Choose your data source: Real market data from YFinance (rate-limited), or generated mock data (unlimited). 1m data is limited to ~7 days, 5m data up to ~60 days."
                    />
                    
                    {dataExpanded && (
                        <div>
                            <label style={{ fontSize: '12px', display: 'block', marginBottom: '5px' }}>Symbol</label>
                            <input 
                                value={symbol} 
                                onChange={e => setSymbol(e.target.value)} 
                                style={{ width: '100%', padding: '6px', background: '#333', border: '1px solid #555', color: '#fff', borderRadius: '3px' }}
                                placeholder="MES=F, ES=F, AAPL"
                            />

                            <label style={{ marginTop: '10px', display: 'block', fontSize: '12px', marginBottom: '5px' }}>Source Granularity</label>
                            <div style={{ display: 'flex', gap: '5px' }}>
                                <button
                                    onClick={() => setSourceInterval('1m')}
                                    style={{ 
                                        flex: 1, 
                                        padding: '6px', 
                                        background: sourceInterval === '1m' ? '#007acc' : '#444', 
                                        border: 'none', 
                                        color: 'white',
                                        borderRadius: '3px',
                                        cursor: 'pointer',
                                        fontSize: '12px'
                                    }}>
                                    1 Minute
                                </button>
                                <button
                                    onClick={() => setSourceInterval('5m')}
                                    style={{ 
                                        flex: 1, 
                                        padding: '6px', 
                                        background: sourceInterval === '5m' ? '#007acc' : '#444', 
                                        border: 'none', 
                                        color: 'white',
                                        borderRadius: '3px',
                                        cursor: 'pointer',
                                        fontSize: '12px'
                                    }}>
                                    5 Minute
                                </button>
                            </div>

                            <label style={{ marginTop: '10px', display: 'block', fontSize: '12px', marginBottom: '5px' }}>Days to Load</label>
                            <input 
                                type="number" 
                                value={loadDays} 
                                onChange={e => setLoadDays(Number(e.target.value))} 
                                min={1} 
                                max={60} 
                                style={{ width: '100%', padding: '6px', background: '#333', border: '1px solid #555', color: '#fff', borderRadius: '3px' }}
                            />
                            <div style={{ fontSize: '10px', color: '#777', marginTop: '2px' }}>Max: ~7d (1m), ~60d (5m)</div>

                            <label style={{ marginTop: '10px', display: 'flex', alignItems: 'center', gap: '5px', cursor: 'pointer' }}>
                                <input
                                    type="checkbox"
                                    checked={useMockData}
                                    onChange={e => setUseMockData(e.target.checked)}
                                    style={{ cursor: 'pointer' }}
                                />
                                <span style={{ fontSize: '12px' }}>Use Mock Data (testing)</span>
                            </label>

                            <button 
                                onClick={loadData} 
                                disabled={isLoading} 
                                style={{ 
                                    width: '100%', 
                                    marginTop: '12px', 
                                    padding: '8px', 
                                    background: isLoading ? '#555' : '#007acc', 
                                    color: 'white', 
                                    border: 'none', 
                                    cursor: isLoading ? 'default' : 'pointer',
                                    borderRadius: '3px',
                                    fontWeight: 'bold',
                                    fontSize: '13px'
                                }}>
                                {isLoading ? 'Loading...' : 'Load Data'}
                            </button>
                            
                            {allData.length > 0 && (
                                <div style={{ marginTop: '8px', fontSize: '11px', color: '#888', textAlign: 'center' }}>
                                    ✓ Loaded {allData.length} bars
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Model Configuration Section */}
                <div style={{ marginBottom: '20px' }}>
                    <SectionHeader 
                        title="Model Configuration" 
                        expanded={modelExpanded} 
                        onToggle={() => setModelExpanded(!modelExpanded)}
                        helpKey="modelConfig"
                        helpText="Select the trained CNN model and chart display settings."
                    />
                    <HelpText 
                        helpKey="modelConfig" 
                        text="CNN models are trained to detect specific setups. Predictive models work for mean reversion. Chart timeframe is for display only and doesn't affect model analysis."
                    />
                    
                    {modelExpanded && (
                        <div>
                            <label style={{ fontSize: '12px', display: 'block', marginBottom: '5px' }}>Model</label>
                            <select 
                                value={selectedModel} 
                                onChange={e => setSelectedModel(e.target.value)} 
                                style={{ width: '100%', padding: '6px', background: '#333', border: '1px solid #555', color: '#fff', borderRadius: '3px' }}
                            >
                                {availableModels.map(m => <option key={m}>{m}</option>)}
                            </select>
                            
                            <label style={{ marginTop: '10px', fontSize: '12px', display: 'block', marginBottom: '5px' }}>Chart Timeframe (Display)</label>
                            <select 
                                value={displayTimeframe} 
                                onChange={e => {
                                    setDisplayTimeframe(Number(e.target.value));
                                    setChartCandles([]);
                                }} 
                                style={{ width: '100%', padding: '6px', background: '#333', border: '1px solid #555', color: '#fff', borderRadius: '3px' }}
                            >
                                <option value={5} disabled={sourceInterval === '5m'}>5 Minutes</option>
                                <option value={15}>15 Minutes</option>
                                <option value={60}>1 Hour</option>
                            </select>
                            <div style={{ fontSize: '10px', color: '#777', marginTop: '2px' }}>For visualization only</div>
                        </div>
                    )}
                </div>

                {/* Entry Mechanism Section */}
                <div style={{ marginBottom: '20px' }}>
                    <SectionHeader 
                        title="Entry Mechanism" 
                        expanded={entryExpanded} 
                        onToggle={() => setEntryExpanded(!entryExpanded)}
                        helpKey="entryMech"
                        helpText="Configure how trades are entered when signals fire."
                    />
                    <HelpText 
                        helpKey="entryMech" 
                        text="Sensitivity: Lower = more selective (fewer trades). Predictive Limit: Places OCO bracket orders at ±N×ATR. First to fill cancels the other. 15min expiry."
                    />

                    {entryExpanded && (
                        <div>
                            <label style={{ fontSize: '12px', display: 'block', marginBottom: '5px' }}>
                                Sensitivity: {(modelThreshold * 100).toFixed(0)}% 
                                <span style={{ fontSize: '10px', color: '#888', marginLeft: '5px' }}>
                                    (min confidence)
                                </span>
                            </label>
                            <input
                                type="range" min="0" max="100" step="1"
                                value={modelThreshold * 100}
                                onChange={e => setModelThreshold(Number(e.target.value) / 100)}
                                style={{ width: '100%' }}
                            />
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#777' }}>
                                <span>Aggressive</span>
                                <span>Conservative</span>
                            </div>

                            <label style={{ display: 'block', marginTop: '10px', fontSize: '12px', marginBottom: '5px' }}>Method</label>
                            <select 
                                value={entryMechanism} 
                                onChange={e => setEntryMechanism(e.target.value as any)} 
                                style={{ width: '100%', padding: '6px', background: '#333', border: '1px solid #555', color: '#fff', borderRadius: '3px' }}
                            >
                                <option>Predictive Limit</option>
                                <option disabled>Market Close (Coming Soon)</option>
                            </select>

                            {entryMechanism === 'Predictive Limit' && (
                                <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                                    <div style={{ flex: 1 }}>
                                        <label style={{ fontSize: '11px', display: 'block', marginBottom: '3px' }}>Limit (×ATR)</label>
                                        <input 
                                            type="number" 
                                            step="0.1" 
                                            value={limitFactor} 
                                            onChange={e => setLimitFactor(Number(e.target.value))} 
                                            style={{ width: '100%', padding: '4px', background: '#333', border: '1px solid #555', color: '#fff', borderRadius: '3px', fontSize: '12px' }}
                                        />
                                    </div>
                                    <div style={{ flex: 1 }}>
                                        <label style={{ fontSize: '11px', display: 'block', marginBottom: '3px' }}>Stop (×ATR)</label>
                                        <input 
                                            type="number" 
                                            step="0.1" 
                                            value={slFactor} 
                                            onChange={e => setSlFactor(Number(e.target.value))} 
                                            style={{ width: '100%', padding: '4px', background: '#333', border: '1px solid #555', color: '#fff', borderRadius: '3px', fontSize: '12px' }}
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Performance Stats Section */}
                <div>
                    <SectionHeader 
                        title="Performance" 
                        expanded={statsExpanded} 
                        onToggle={() => setStatsExpanded(!statsExpanded)}
                    />
                    
                    {statsExpanded && (
                        <div>
                            <div style={{ background: '#333', padding: '12px', borderRadius: '4px', marginBottom: '10px' }}>
                                <div style={{ marginBottom: '8px' }}>
                                    <div style={{ fontSize: '11px', color: '#888' }}>Realized PnL</div>
                                    <div style={{ fontSize: '18px', fontWeight: 'bold', color: totalPnL >= 0 ? '#4caf50' : '#f44336' }}>
                                        ${totalPnL.toFixed(2)}
                                    </div>
                                </div>
                                <div style={{ marginBottom: '8px' }}>
                                    <div style={{ fontSize: '11px', color: '#888' }}>Floating PnL</div>
                                    <div style={{ fontSize: '16px', fontWeight: 'bold', color: unrealizedPnL >= 0 ? '#4caf50' : '#f44336' }}>
                                        ${unrealizedPnL.toFixed(2)}
                                    </div>
                                </div>
                                <div style={{ paddingTop: '8px', borderTop: '1px solid #444', fontSize: '12px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                        <span>Open Positions:</span>
                                        <strong>{openTrades.length}</strong>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                        <span>Pending Orders:</span>
                                        <strong>{pendingTrades.length}</strong>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <span>Closed Trades:</span>
                                        <strong>{closedTrades.length}</strong>
                                    </div>
                                </div>
                            </div>
                            
                            {closedTrades.length > 0 && (
                                <div style={{ background: '#2a2a2a', padding: '8px', borderRadius: '4px', fontSize: '11px' }}>
                                    <div style={{ marginBottom: '4px' }}>
                                        Win Rate: <strong>
                                            {((closedTrades.filter(t => (t.pnl ?? 0) > 0).length / closedTrades.length) * 100).toFixed(1)}%
                                        </strong>
                                    </div>
                                    <div>
                                        Avg PnL: <strong style={{ color: (totalPnL / closedTrades.length) >= 0 ? '#4caf50' : '#f44336' }}>
                                            ${(totalPnL / closedTrades.length).toFixed(2)}
                                        </strong>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {/* Main Chart */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                {/* Control Bar */}
                <div style={{ 
                    padding: '12px 15px', 
                    display: 'flex', 
                    gap: '15px', 
                    alignItems: 'center', 
                    background: '#2d2d2d',
                    borderBottom: '1px solid #444'
                }}>
                    <button 
                        onClick={() => setIsPlaying(!isPlaying)} 
                        style={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: '5px', 
                            padding: '8px 16px', 
                            cursor: 'pointer',
                            background: isPlaying ? '#f44336' : '#4caf50',
                            border: 'none',
                            color: 'white',
                            borderRadius: '4px',
                            fontWeight: 'bold',
                            fontSize: '13px'
                        }}
                        disabled={allData.length === 0}
                    >
                        {isPlaying ? <><Pause size={16} /> Pause</> : <><Play size={16} /> Play</>}
                    </button>
                    <button 
                        onClick={() => resetPlayback()} 
                        style={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: '5px', 
                            padding: '8px 16px', 
                            cursor: 'pointer',
                            background: '#555',
                            border: 'none',
                            color: 'white',
                            borderRadius: '4px',
                            fontSize: '13px'
                        }}
                        disabled={allData.length === 0}
                    >
                        <RotateCcw size={16} /> Reset
                    </button>
                    
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginLeft: '10px' }}>
                        <span style={{ fontSize: '12px', color: '#aaa' }}>Speed:</span>
                        <input 
                            type="range" 
                            min="10" 
                            max="1000" 
                            step="10" 
                            value={playbackSpeed} 
                            onChange={e => setPlaybackSpeed(Number(e.target.value))}
                            style={{ width: '120px' }}
                        />
                        <span style={{ fontSize: '12px', color: '#ddd', minWidth: '60px' }}>{playbackSpeed}ms</span>
                    </div>
                    
                    <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '15px' }}>
                        <div style={{ fontSize: '12px', color: '#aaa' }}>
                            Progress: <strong style={{ color: '#fff' }}>
                                {allData.length > 0 ? `${currentIndex} / ${allData.length}` : '--'}
                            </strong>
                        </div>
                        <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#fff' }}>
                            {allData[currentIndex]?.time ? new Date(allData[currentIndex].time * 1000).toLocaleString() : '--'}
                        </div>
                    </div>
                </div>

                {/* Signal Indicator */}
                <div style={{ 
                    position: 'absolute', 
                    bottom: '20px', 
                    left: '340px', 
                    background: 'rgba(0,0,0,0.85)', 
                    padding: '15px', 
                    borderRadius: '8px', 
                    display: 'flex', 
                    gap: '20px', 
                    alignItems: 'center', 
                    zIndex: 100, 
                    border: '1px solid #555',
                    backdropFilter: 'blur(10px)'
                }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '5px' }}>
                        <div style={{
                            width: '40px', 
                            height: '40px', 
                            borderRadius: '50%',
                            background: lastAnalysis && lastAnalysis.prob > 0 
                                ? `radial-gradient(circle, rgba(0, 255, 0, ${lastAnalysis.prob}), rgba(0, 128, 0, ${lastAnalysis.prob * 0.5}))` 
                                : '#333',
                            border: '2px solid #555',
                            boxShadow: lastAnalysis && lastAnalysis.prob > modelThreshold 
                                ? `0 0 20px rgba(0,255,0,${lastAnalysis.prob})` 
                                : 'none',
                            transition: 'all 0.3s ease'
                        }} />
                        <div style={{ fontSize: '9px', color: '#888', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                            Signal
                        </div>
                    </div>
                    
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <div>
                            <div style={{ fontSize: '10px', color: '#aaa', marginBottom: '2px' }}>Model Confidence</div>
                            <div style={{ 
                                fontSize: '18px', 
                                fontWeight: 'bold', 
                                color: lastAnalysis && lastAnalysis.prob > modelThreshold ? '#4caf50' : '#fff'
                            }}>
                                {lastAnalysis ? `${(lastAnalysis.prob * 100).toFixed(1)}%` : '--'}
                            </div>
                        </div>
                        
                        <div>
                            <div style={{ fontSize: '10px', color: '#aaa', marginBottom: '2px' }}>ATR (15m)</div>
                            <div style={{ fontSize: '14px', color: '#ddd', fontWeight: 'bold' }}>
                                {lastAnalysis ? lastAnalysis.atr.toFixed(2) : '--'}
                            </div>
                        </div>
                    </div>
                    
                    {lastAnalysis && lastAnalysis.prob > modelThreshold && (
                        <div style={{
                            padding: '4px 8px',
                            background: 'rgba(76, 175, 80, 0.2)',
                            border: '1px solid #4caf50',
                            borderRadius: '4px',
                            fontSize: '11px',
                            color: '#4caf50',
                            fontWeight: 'bold',
                            animation: 'pulse 2s infinite'
                        }}>
                            ACTIVE
                        </div>
                    )}
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
