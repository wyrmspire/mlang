import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, ColorType } from 'lightweight-charts';
import { Candle } from '../api/client';

interface ChartPanelProps {
    data: Candle[];
    syntheticData?: Candle[];
    title?: string;
}

export const ChartPanel: React.FC<ChartPanelProps> = ({ data, syntheticData, title }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    const syntheticSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

    useEffect(() => {
        if (!containerRef.current) return;

        const chart = createChart(containerRef.current, {
            width: containerRef.current.clientWidth,
            height: containerRef.current.clientHeight, // Use actual height
            layout: {
                background: { type: ColorType.Solid, color: '#1E1E1E' },
                textColor: '#DDD',
            },
            grid: {
                vertLines: { color: '#333' },
                horzLines: { color: '#333' },
            },
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            }
        });

        // ... (series creation remains same)
        const series = chart.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        });

        const syntheticSeries = chart.addCandlestickSeries({
            upColor: '#4caf50',
            downColor: '#ff9800',
            borderVisible: true,
            borderColor: '#ffff00',
            wickUpColor: '#4caf50',
            wickDownColor: '#ff9800',
        });

        chartRef.current = chart;
        seriesRef.current = series;
        syntheticSeriesRef.current = syntheticSeries;

        // Resize Observer
        const resizeObserver = new ResizeObserver(entries => {
            if (!entries || entries.length === 0) return;
            const { width, height } = entries[0].contentRect;
            chart.applyOptions({ width, height });
            chart.timeScale().fitContent();
        });

        resizeObserver.observe(containerRef.current);

        return () => {
            resizeObserver.disconnect();
            chart.remove();
        };
    }, []);

    // ... (rest of effects)

    useEffect(() => {
        if (seriesRef.current && data.length > 0) {
            // Sort just in case
            const sorted = [...data].sort((a, b) => a.time - b.time);
            seriesRef.current.setData(sorted as any);
            chartRef.current?.timeScale().fitContent();
        } else if (seriesRef.current) {
            seriesRef.current.setData([]);
        }
    }, [data]);

    useEffect(() => {
        if (syntheticSeriesRef.current) {
            if (syntheticData && syntheticData.length > 0) {
                const sorted = [...syntheticData].sort((a, b) => a.time - b.time);
                syntheticSeriesRef.current.setData(sorted as any);
            } else {
                syntheticSeriesRef.current.setData([]);
            }
        }
    }, [syntheticData]);

    return (
        <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
            {title && <div style={{ padding: '5px', background: '#333', fontSize: '12px' }}>{title}</div>}
            <div ref={containerRef} style={{ width: '100%', flex: 1, minHeight: '0', overflow: 'hidden' }} />
        </div>
    );
};
