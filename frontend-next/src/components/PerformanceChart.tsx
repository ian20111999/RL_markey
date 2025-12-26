'use client';

import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from 'recharts';

interface ChartDataPoint {
    date: string;
    pnl: number;
    trades: number;
}

interface PerformanceChartProps {
    data: ChartDataPoint[];
}

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
        return (
            <div className="glass rounded-lg p-3 border border-white/10">
                <p className="text-sm text-gray-400 mb-1">{label}</p>
                <p className="text-lg font-semibold text-emerald-400">
                    ${payload[0].value.toLocaleString()}
                </p>
                {payload[1] && (
                    <p className="text-xs text-gray-500 mt-1">
                        交易次數: {payload[1].value}
                    </p>
                )}
            </div>
        );
    }
    return null;
};

export default function PerformanceChart({ data }: PerformanceChartProps) {
    return (
        <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                    data={data}
                    margin={{ top: 10, right: 10, left: -10, bottom: 0 }}
                >
                    <defs>
                        <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                    <XAxis
                        dataKey="date"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: '#71717a', fontSize: 12 }}
                    />
                    <YAxis
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: '#71717a', fontSize: 12 }}
                        tickFormatter={(value) => `$${value}`}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                        type="monotone"
                        dataKey="pnl"
                        stroke="#6366f1"
                        strokeWidth={2}
                        fill="url(#pnlGradient)"
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
}
