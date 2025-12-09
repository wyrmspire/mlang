import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, ColorType } from 'lightweight-charts';
import { Candle } from '../api/client';
import { Trade } from '../api/yfinance';

interface YFinanceChartProps {
    data: Candle[];
    trades: Trade[];
    currentPrice: number;
}

export const YFinanceChart: React.FC<YFinanceChartProps> = ({ data, trades, currentPrice }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    const markersRef = useRef<any[]>([]);

    useEffect(() => {
        if (!containerRef.current) return;

        const chart = createChart(containerRef.current, {
            width: containerRef.current.clientWidth,
            height: containerRef.current.clientHeight,
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
            },
            crosshair: {
                mode: 1,
            }
        });

        const series = chart.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        });

        chartRef.current = chart;
        seriesRef.current = series;

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

    // Update candle data
    useEffect(() => {
        if (seriesRef.current && data.length > 0) {
            const sorted = [...data].sort((a, b) => a.time - b.time);
            seriesRef.current.setData(sorted as any);
            
            // Auto-scroll to latest
            if (chartRef.current) {
                chartRef.current.timeScale().scrollToRealTime();
            }
        }
    }, [data]);

    // Update markers for trades
    useEffect(() => {
        if (!seriesRef.current) return;

        const markers: any[] = [];

        trades.forEach(trade => {
            // Entry marker
            markers.push({
                time: trade.entry_time,
                position: trade.direction === 'LONG' ? 'belowBar' : 'aboveBar',
                color: trade.direction === 'LONG' ? '#26a69a' : '#ef5350',
                shape: 'arrowUp',
                text: `${trade.direction} @ ${trade.entry_price.toFixed(2)}`
            });

            // Exit marker (if closed)
            if (trade.status === 'closed' && trade.exit_time && trade.exit_price) {
                const isWin = (trade.pnl || 0) > 0;
                markers.push({
                    time: trade.exit_time,
                    position: trade.direction === 'LONG' ? 'aboveBar' : 'belowBar',
                    color: isWin ? '#4caf50' : '#ff5252',
                    shape: 'arrowDown',
                    text: `Exit @ ${trade.exit_price.toFixed(2)} | ${isWin ? 'WIN' : 'LOSS'} $${Math.abs(trade.pnl || 0).toFixed(0)}`
                });
            }
        });

        seriesRef.current.setMarkers(markers);
        markersRef.current = markers;
    }, [trades]);

    // Draw SL/TP lines for open trades
    useEffect(() => {
        if (!chartRef.current || !data.length) return;

        // Note: lightweight-charts doesn't have easy way to remove all lines,
        // so we'll rely on component remount for now
        // Lines are shown in the sidebar and overlays which is sufficient
        
    }, [trades, data]);

    return (
        <div style={{ width: '100%', height: '100%', position: 'relative' }}>
            <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
            
            {/* Current Price Overlay */}
            {currentPrice > 0 && (
                <div style={{
                    position: 'absolute',
                    top: '10px',
                    right: '10px',
                    background: 'rgba(0, 0, 0, 0.7)',
                    padding: '8px 12px',
                    borderRadius: '4px',
                    fontSize: '14px',
                    fontWeight: 'bold',
                    color: '#4caf50'
                }}>
                    Current: ${currentPrice.toFixed(2)}
                </div>
            )}

            {/* Active Trades Overlay */}
            {trades.filter(t => t.status === 'open').length > 0 && (
                <div style={{
                    position: 'absolute',
                    top: '50px',
                    right: '10px',
                    background: 'rgba(0, 0, 0, 0.7)',
                    padding: '8px',
                    borderRadius: '4px',
                    fontSize: '11px',
                    maxWidth: '200px'
                }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Active Positions</div>
                    {trades.filter(t => t.status === 'open').map(trade => (
                        <div key={trade.id} style={{ 
                            marginBottom: '5px', 
                            paddingBottom: '5px', 
                            borderBottom: '1px solid #444' 
                        }}>
                            <div style={{ color: trade.direction === 'LONG' ? '#26a69a' : '#ef5350' }}>
                                {trade.direction}
                            </div>
                            <div>Entry: {trade.entry_price.toFixed(2)}</div>
                            <div style={{ fontSize: '10px', color: '#999' }}>
                                SL: {trade.sl_price.toFixed(2)} | TP: {trade.tp_price.toFixed(2)}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
