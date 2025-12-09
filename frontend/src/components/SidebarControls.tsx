import React from 'react';

interface SidebarControlsProps {
    dates: string[];
    selectedDate: string;
    onDateChange: (d: string) => void;
    timeframe: string;
    onTimeframeChange: (t: string) => void;
    onGenerate: () => void;
    isGenerating: boolean;
    dayOfWeek: number;
    setDayOfWeek: (d: number) => void;
    sessionType: string;
    setSessionType: (s: string) => void;
    onClearSynthetic: () => void;
    onOpenYFinance?: () => void;

    // Multi-Day Props
    isMultiDay: boolean;
    setIsMultiDay: (b: boolean) => void;
    numDays: number;
    setNumDays: (n: number) => void;
    startDate: string;         // For multi-day range start
    setStartDate: (d: string) => void;
}

export const SidebarControls: React.FC<SidebarControlsProps> = ({
    dates, selectedDate, onDateChange,
    timeframe, onTimeframeChange,
    onGenerate, isGenerating,
    dayOfWeek, setDayOfWeek,
    sessionType, setSessionType,
    onClearSynthetic,
    onOpenYFinance,
    isMultiDay, setIsMultiDay,
    numDays, setNumDays,
    startDate, setStartDate
}) => {
    return (
        <div style={{ width: '250px', background: '#252526', padding: '10px', borderRight: '1px solid #333' }}>
            <h3>Data Controls</h3>

            <div style={{ marginBottom: '15px' }}>
                <label style={{ display: 'block', marginBottom: '5px' }}>Mode</label>
                <div style={{ display: 'flex', gap: '10px' }}>
                    <button
                        onClick={() => setIsMultiDay(false)}
                        style={{ flex: 1, background: !isMultiDay ? '#007ACC' : '#444', color: '#FFF', border: 'none', padding: '5px' }}
                    >
                        Single
                    </button>
                    <button
                        onClick={() => setIsMultiDay(true)}
                        style={{ flex: 1, background: isMultiDay ? '#007ACC' : '#444', color: '#FFF', border: 'none', padding: '5px' }}
                    >
                        Multi-Day
                    </button>
                </div>
            </div>

            {!isMultiDay ? (
                // SINGLE DAY CONTROLS
                <>
                    <div style={{ marginBottom: '15px' }}>
                        <label style={{ display: 'block', marginBottom: '5px' }}>Date</label>
                        <select
                            value={selectedDate}
                            onChange={e => onDateChange(e.target.value)}
                            style={{ width: '100%', padding: '5px', background: '#333', color: '#EEE', border: '1px solid #555' }}
                        >
                            {dates.map(d => <option key={d} value={d}>{d}</option>)}
                        </select>
                    </div>
                </>
            ) : (
                // MULTI DAY CONTROLS
                <>
                    <div style={{ marginBottom: '15px' }}>
                        <label style={{ display: 'block', marginBottom: '5px' }}>Start Date</label>
                        <select
                            value={startDate}
                            onChange={e => setStartDate(e.target.value)}
                            style={{ width: '100%', padding: '5px', background: '#333', color: '#EEE', border: '1px solid #555' }}
                        >
                            {dates.map(d => <option key={d} value={d}>{d}</option>)}
                        </select>
                    </div>
                    <div style={{ marginBottom: '15px' }}>
                        <label style={{ display: 'block', marginBottom: '5px' }}>Num Days</label>
                        <input
                            type="number"
                            value={numDays}
                            onChange={e => setNumDays(Number(e.target.value))}
                            min={2} max={90}
                            style={{ width: '100%', padding: '5px', background: '#333', color: '#EEE', border: '1px solid #555' }}
                        />
                    </div>
                </>
            )}

            <div style={{ marginBottom: '15px' }}>
                <label style={{ display: 'block', marginBottom: '5px' }}>Timeframe</label>
                <select
                    value={timeframe}
                    onChange={e => onTimeframeChange(e.target.value)}
                    style={{ width: '100%', padding: '5px', background: '#333', color: '#EEE', border: '1px solid #555' }}
                >
                    <option value="1m">1 Minute</option>
                    <option value="5m">5 Minutes</option>
                    <option value="15m">15 Minutes</option>
                    <option value="30m">30 Minutes</option>
                    <option value="60m">1 Hour</option>
                </select>
            </div>

            <hr style={{ borderColor: '#444', margin: '15px 0' }} />

            <h3>Generator</h3>

            <div style={{ marginBottom: '10px' }}>
                <label style={{ display: 'block', marginBottom: '5px' }}>Session Type</label>
                <select value={sessionType} onChange={e => setSessionType(e.target.value)} style={{ width: '100%', padding: '5px', background: '#333', color: '#EEE', border: '1px solid #555' }}>
                    <option value="RTH">RTH (08:30-15:15)</option>
                    <option value="overnight">Overnight (17:00-08:30)</option>
                </select>
            </div>

            {!isMultiDay && (
                <div style={{ marginBottom: '10px' }}>
                    <label style={{ display: 'block', marginBottom: '5px' }}>Day of Week</label>
                    <select value={dayOfWeek} onChange={e => setDayOfWeek(Number(e.target.value))} style={{ width: '100%', padding: '5px', background: '#333', color: '#EEE', border: '1px solid #555' }}>
                        <option value={0}>Monday</option>
                        <option value={1}>Tuesday</option>
                        <option value={2}>Wednesday</option>
                        <option value={3}>Thursday</option>
                        <option value={4}>Friday</option>
                        <option value={5}>Saturday</option>
                        <option value={6}>Sunday</option>
                    </select>
                </div>
            )}

            <button
                onClick={onGenerate}
                disabled={isGenerating}
                style={{
                    width: '100%', padding: '10px',
                    background: isGenerating ? '#555' : '#0E639C',
                    color: 'white', border: 'none', cursor: isGenerating ? 'wait' : 'pointer',
                    marginTop: '10px'
                }}
            >
                {isGenerating ? 'Generating...' : isMultiDay ? 'Generate Sequence' : 'Generate Session'}
            </button>

            <button
                onClick={onClearSynthetic}
                style={{
                    width: '100%', padding: '10px',
                    background: '#333',
                    color: '#CCC', border: '1px solid #555', cursor: 'pointer',
                    marginTop: '5px'
                }}
            >
                Clear Synthetic
            </button>

            {onOpenYFinance && (
                <button
                    onClick={onOpenYFinance}
                    style={{
                        width: '100%', padding: '10px',
                        background: '#D4A500',
                        color: '#000', border: 'none', cursor: 'pointer',
                        marginTop: '10px',
                        fontWeight: 'bold'
                    }}
                >
                    📈 YFinance Data
                </button>
            )}
        </div>
    );
};
