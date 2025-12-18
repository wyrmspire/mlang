import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, Time } from 'lightweight-charts';
import { VizDecision, VizTrade } from '../types/viz';

interface CandleChartProps {
    decision: VizDecision | null;
    trade: VizTrade | null;
}

type Timeframe = '1m' | '5m' | '15m';

// Aggregation helper: turns 1m candles into Xm candles
// data: [open, high, low, close, volume]
const aggregateData = (data: number[][], interval: number): number[][] => {
    if (interval === 1) return data;

    const aggregated: number[][] = [];

    // Iterate backwards to anchor on the most recent data point (decision time)
    // This ensures the last bar on the chart aligns with the decision regardless of total count
    for (let i = data.length - 1; i >= 0; i -= interval) {
        const chunk: number[][] = [];
        // Collect up to 'interval' candles for this period
        for (let j = 0; j < interval; j++) {
            const idx = i - j;
            if (idx >= 0) {
                // We unshift to keep them chronological [oldest ... newest] within the chunk
                chunk.unshift(data[idx]);
            }
        }

        if (chunk.length === 0) continue;

        const open = chunk[0][0]; // Open of the first candle in chunk
        const close = chunk[chunk.length - 1][3]; // Close of the last candle in chunk
        let high = -Infinity;
        let low = Infinity;
        let vol = 0;

        chunk.forEach(c => {
            if (c[1] > high) high = c[1];
            if (c[2] < low) low = c[2];
            vol += c[4];
        });

        // Unshift to result to maintain [oldest ... newest] order for the final array
        aggregated.unshift([open, high, low, close, vol]);
    }
    return aggregated;
};

export const CandleChart: React.FC<CandleChartProps> = ({ decision, trade }) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

    // References for OCO lines
    const entryLineRef = useRef<any>(null);
    const stopLineRef = useRef<any>(null);
    const tpLineRef = useRef<any>(null);

    const [timeframe, setTimeframe] = useState<Timeframe>('1m');

    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: '#1e293b' },
                textColor: '#cbd5e1',
            },
            grid: {
                vertLines: { color: '#334155' },
                horzLines: { color: '#334155' },
            },
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            },
        });

        const newSeries = chart.addCandlestickSeries({
            upColor: '#22c55e',
            downColor: '#ef4444',
            borderVisible: false,
            wickUpColor: '#22c55e',
            wickDownColor: '#ef4444',
        });

        chartRef.current = chart;
        seriesRef.current = newSeries;

        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, []);

    // Update Data when decision or timeframe changes
    useEffect(() => {
        if (!seriesRef.current || !decision) return;

        // Handle both formats: decision.window.x_price_1m (viz export) and decision.x_price_1m (OR export)
        const rawData = decision.window?.x_price_1m || (decision as any).x_price_1m;

        if (!rawData || rawData.length === 0) return;

        // Determine aggregation interval
        const intervalMap = { '1m': 1, '5m': 5, '15m': 15 };
        const interval = intervalMap[timeframe];

        const processedData = aggregateData(rawData, interval);

        // Calculate Timestamps
        // We assume the last data point corresponds to the decision timestamp.
        // Each preceding data point is 'interval' minutes before.
        const baseTime = new Date(decision.timestamp || Date.now()).getTime() / 1000;

        const chartData = processedData.map((d, i) => {
            const offset = processedData.length - 1 - i;
            return {
                time: (baseTime - (offset * interval * 60)) as Time,
                open: d[0],
                high: d[1],
                low: d[2],
                close: d[3],
            };
        });

        seriesRef.current.setData(chartData);

        // Remove old lines
        if (entryLineRef.current) { seriesRef.current.removePriceLine(entryLineRef.current); entryLineRef.current = null; }
        if (stopLineRef.current) { seriesRef.current.removePriceLine(stopLineRef.current); stopLineRef.current = null; }
        if (tpLineRef.current) { seriesRef.current.removePriceLine(tpLineRef.current); tpLineRef.current = null; }

        // Add OCO Lines if present
        if (decision.oco) {
            entryLineRef.current = seriesRef.current.createPriceLine({
                price: decision.oco.entry_price,
                color: '#3b82f6', // blue
                lineWidth: 2,
                lineStyle: 0, // Solid
                axisLabelVisible: true,
                title: 'ENTRY',
            });

            stopLineRef.current = seriesRef.current.createPriceLine({
                price: decision.oco.stop_price,
                color: '#ef4444', // red
                lineWidth: 2,
                lineStyle: 2, // Dashed
                axisLabelVisible: true,
                title: 'STOP',
            });

            tpLineRef.current = seriesRef.current.createPriceLine({
                price: decision.oco.tp_price,
                color: '#22c55e', // green
                lineWidth: 2,
                lineStyle: 2, // Dashed
                axisLabelVisible: true,
                title: 'TP',
            });
        }

        if (chartRef.current) {
            chartRef.current.timeScale().fitContent();
        }

    }, [decision, timeframe]);

    return (
        <div className="relative w-full h-full group">
            <div ref={chartContainerRef} className="w-full h-full" />

            {/* Timeframe Controls */}
            <div className="absolute top-3 right-3 flex bg-slate-800 rounded-md border border-slate-700 shadow-lg overflow-hidden z-10">
                {(['1m', '5m', '15m'] as Timeframe[]).map((tf) => (
                    <button
                        key={tf}
                        onClick={() => setTimeframe(tf)}
                        className={`px-3 py-1 text-xs font-bold transition-colors ${timeframe === tf
                                ? 'bg-blue-600 text-white'
                                : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
                            } ${tf !== '15m' ? 'border-r border-slate-700' : ''}`}
                    >
                        {tf}
                    </button>
                ))}
            </div>
        </div>
    );
};
