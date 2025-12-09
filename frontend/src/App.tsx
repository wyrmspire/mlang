import React, { useState, useEffect } from 'react';
import { SidebarControls } from './components/SidebarControls';
import { ChartPanel } from './components/ChartPanel';
import { YFinanceMode } from './components/YFinanceMode';
import { api, Candle } from './api/client';

type AppMode = 'pattern' | 'yfinance';

const App: React.FC = () => {
    // Mode selection
    const [mode, setMode] = useState<AppMode>('pattern');

    // Data State
    const [dates, setDates] = useState<string[]>([]);

    // Single Day State
    const [selectedDate, setSelectedDate] = useState<string>('');
    const [candles, setCandles] = useState<Candle[]>([]);
    const [syntheticData, setSyntheticData] = useState<Candle[]>([]); // Single day overlay

    // Multi Day State
    const [isMultiDay, setIsMultiDay] = useState(false);
    const [startDate, setStartDate] = useState<string>('');
    const [numDays, setNumDays] = useState<number>(5);
    const [multiDayReal, setMultiDayReal] = useState<Candle[]>([]);
    const [multiDaySynth, setMultiDaySynth] = useState<Candle[]>([]);

    // UI State
    const [timeframe, setTimeframe] = useState<string>('5m');
    const [isGenerating, setIsGenerating] = useState(false);

    // Gen Params
    const [genDayOfWeek, setGenDayOfWeek] = useState<number>(0);
    const [genSessionType, setGenSessionType] = useState<string>('RTH');

    // Initial Load
    useEffect(() => {
        if (mode === 'pattern') {
            loadDates();
        }
    }, [mode]);

    const loadDates = async () => {
        try {
            const d = await api.getDates();
            setDates(d);
            if (d.length > 0) {
                // Default to newest for Single Day
                setSelectedDate(d[0]);

                // Default to ~30 days back for Multi-Day so we have forward history
                const offset = Math.min(30, d.length - 1);
                setStartDate(d[offset]);
            }
        } catch (e) {
            console.error(e);
        }
    };

    // Load Data Effect
    useEffect(() => {
        if (!selectedDate && !startDate && dates.length === 0) return;

        const fetchData = async () => {
            try {
                if (!isMultiDay) {
                    // Single Day Mode
                    if (selectedDate) {
                        const data = await api.getCandles(selectedDate, timeframe);
                        setCandles(data);
                    }
                } else {
                    // Multi Day Mode
                    if (startDate && dates.length > 0) {
                        const startIdx = dates.indexOf(startDate);
                        if (startIdx >= 0) {
                            // Calculate "Future" End Date (chronologically later => smaller index)
                            let endIdx = startIdx - numDays + 1;
                            if (endIdx < 0) endIdx = 0; // Clamp to newest available

                            const endDate = dates[endIdx];

                            if (startIdx >= endIdx) {
                                const data = await api.getCandlesRange(startDate, endDate, timeframe);
                                setMultiDayReal(data);
                            }
                        }
                    }
                }
            } catch (e) {
                console.error(e);
            }
        };

        fetchData();

        // Auto-set generator Day of Week to match selected date (Single Mode only)
        if (!isMultiDay && selectedDate) {
            try {
                const [y, m, d] = selectedDate.split('-').map(Number);
                const dateObj = new Date(y, m - 1, d);
                let dow = dateObj.getDay();
                dow = dow === 0 ? 6 : dow - 1;
                setGenDayOfWeek(dow);
            } catch (e) { console.error(e); }
        }

    }, [selectedDate, startDate, timeframe, isMultiDay, numDays, dates]);

    // Clear synthetic data when timeframe changes
    useEffect(() => {
        setSyntheticData([]);
        setMultiDaySynth([]);
    }, [timeframe]);


    const handleGenerate = async () => {
        setIsGenerating(true);
        try {
            if (!isMultiDay) {
                // Single Day
                const startPrice = candles.length > 0 ? candles[0].open : 5800;
                const data = await api.generateSession(
                    genDayOfWeek,
                    genSessionType,
                    startPrice,
                    selectedDate,
                    timeframe
                );
                setSyntheticData(data);
            } else {
                // Multi Day
                const startPrice = multiDayReal.length > 0 ? multiDayReal[0].open : 5800;
                const data = await api.generateMultiDay(
                    numDays,
                    genSessionType,
                    startPrice,
                    startDate,
                    timeframe
                );
                setMultiDaySynth(data);
            }
        } catch (e) {
            console.error("Generation failed", e);
            alert("Generation failed. Check console/backend.");
        } finally {
            setIsGenerating(false);
        }
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#1E1E1E', color: '#EEE' }}>
            {/* Top Menu Bar */}
            <div style={{ 
                display: 'flex', 
                gap: '0', 
                background: '#252526', 
                borderBottom: '1px solid #333',
                padding: '0'
            }}>
                <button
                    onClick={() => setMode('pattern')}
                    style={{
                        padding: '15px 30px',
                        background: mode === 'pattern' ? '#1E1E1E' : 'transparent',
                        color: mode === 'pattern' ? '#4FC3F7' : '#AAA',
                        border: 'none',
                        borderBottom: mode === 'pattern' ? '2px solid #4FC3F7' : '2px solid transparent',
                        cursor: 'pointer',
                        fontSize: '14px',
                        fontWeight: mode === 'pattern' ? 'bold' : 'normal',
                        transition: 'all 0.2s'
                    }}
                >
                    Pattern Generator
                </button>
                <button
                    onClick={() => setMode('yfinance')}
                    style={{
                        padding: '15px 30px',
                        background: mode === 'yfinance' ? '#1E1E1E' : 'transparent',
                        color: mode === 'yfinance' ? '#4FC3F7' : '#AAA',
                        border: 'none',
                        borderBottom: mode === 'yfinance' ? '2px solid #4FC3F7' : '2px solid transparent',
                        cursor: 'pointer',
                        fontSize: '14px',
                        fontWeight: mode === 'yfinance' ? 'bold' : 'normal',
                        transition: 'all 0.2s'
                    }}
                >
                    YFinance Playback
                </button>
            </div>

            {/* Content */}
            {mode === 'yfinance' ? (
                <YFinanceMode />
            ) : (
                <div style={{ display: 'flex', height: 'calc(100vh - 52px)' }}>
                    <SidebarControls
                        dates={dates}
                        selectedDate={selectedDate}
                        onDateChange={setSelectedDate}
                        timeframe={timeframe}
                        onTimeframeChange={setTimeframe}
                        onGenerate={handleGenerate}
                        isGenerating={isGenerating}
                        dayOfWeek={genDayOfWeek}
                        setDayOfWeek={setGenDayOfWeek}
                        sessionType={genSessionType}
                        setSessionType={setGenSessionType}
                        onClearSynthetic={() => {
                            setSyntheticData([]);
                            setMultiDaySynth([]);
                        }}

                        // Multi
                        isMultiDay={isMultiDay}
                        setIsMultiDay={setIsMultiDay}
                        numDays={numDays}
                        setNumDays={setNumDays}
                        startDate={startDate}
                        setStartDate={setStartDate}
                    />

                    <div style={{ flex: 1, padding: '10px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        {!isMultiDay ? (
                            // Single Pane with Overlay
                            <div style={{ flex: 1, overflow: 'hidden' }}>
                                <ChartPanel data={candles} syntheticData={syntheticData} title={`Historical: ${selectedDate}`} />
                            </div>
                        ) : (
                            // Split Pane
                            <>
                                <div style={{ flex: 1, border: '1px solid #444', overflow: 'hidden' }}>
                                    <ChartPanel data={multiDayReal} title={`Real Data (${startDate} + ${numDays} days)`} />
                                </div>
                                <div style={{ flex: 1, border: '1px solid #444', overflow: 'hidden' }}>
                                    <ChartPanel data={multiDaySynth} title="Synthetic Generated Path" />
                                </div>
                            </>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default App;
