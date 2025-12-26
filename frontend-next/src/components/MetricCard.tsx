'use client';

import { TrendingUp, TrendingDown } from 'lucide-react';

interface MetricCardProps {
    title: string;
    value: string;
    change?: number;
    subtitle?: string;
    icon: React.ReactNode;
    trend?: 'up' | 'down';
}

export default function MetricCard({
    title,
    value,
    change,
    subtitle,
    icon,
    trend,
}: MetricCardProps) {
    return (
        <div className="metric-card">
            <div className="flex items-start justify-between mb-4">
                <div className="p-2 rounded-lg bg-white/5 text-indigo-400">
                    {icon}
                </div>
                {change !== undefined && (
                    <div
                        className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${trend === 'up'
                                ? 'bg-emerald-500/20 text-emerald-400'
                                : 'bg-red-500/20 text-red-400'
                            }`}
                    >
                        {trend === 'up' ? (
                            <TrendingUp className="w-3 h-3" />
                        ) : (
                            <TrendingDown className="w-3 h-3" />
                        )}
                        {Math.abs(change)}%
                    </div>
                )}
            </div>

            <h3 className="text-sm text-gray-400 mb-1">{title}</h3>
            <p className="text-2xl font-bold">{value}</p>

            {subtitle && (
                <p className="text-xs text-gray-500 mt-2">{subtitle}</p>
            )}
        </div>
    );
}
