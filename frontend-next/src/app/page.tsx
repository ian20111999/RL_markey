'use client';

import { useState, useEffect } from 'react';
import {
    TrendingUp, TrendingDown, Activity, DollarSign,
    BarChart3, Zap, Target, Clock, RefreshCw
} from 'lucide-react';
import Sidebar from '@/components/Sidebar';
import MetricCard from '@/components/MetricCard';
import PerformanceChart from '@/components/PerformanceChart';
import ModelList from '@/components/ModelList';
import TrainingStatus from '@/components/TrainingStatus';

// Mock data
const mockMetrics = {
    totalPnL: 12543.87,
    pnlChange: 5.2,
    sharpeRatio: 2.34,
    sharpeChange: 0.12,
    winRate: 67.5,
    winRateChange: 2.1,
    totalTrades: 1547,
    activeModels: 3,
    maxDrawdown: 8.5,
};

const mockChartData = [
    { date: '12/20', pnl: 1200, trades: 45 },
    { date: '12/21', pnl: 1800, trades: 52 },
    { date: '12/22', pnl: 1500, trades: 48 },
    { date: '12/23', pnl: 2200, trades: 61 },
    { date: '12/24', pnl: 2800, trades: 55 },
    { date: '12/25', pnl: 2400, trades: 49 },
    { date: '12/26', pnl: 3200, trades: 58 },
];

export default function Dashboard() {
    const [isLoading, setIsLoading] = useState(true);
    const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

    useEffect(() => {
        // Simulate data loading
        const timer = setTimeout(() => setIsLoading(false), 1000);
        return () => clearTimeout(timer);
    }, []);

    const handleRefresh = () => {
        setIsLoading(true);
        setLastUpdate(new Date());
        setTimeout(() => setIsLoading(false), 500);
    };

    return (
        <div className="flex min-h-screen">
            <Sidebar />

            <main className="flex-1 p-8 ml-64">
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                    <div>
                        <h1 className="text-3xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
                            Dashboard
                        </h1>
                        <p className="text-gray-500 mt-1">
                            即時監控交易系統表現
                        </p>
                    </div>

                    <div className="flex items-center gap-4">
                        <span className="text-sm text-gray-500">
                            最後更新: {lastUpdate.toLocaleTimeString()}
                        </span>
                        <button
                            onClick={handleRefresh}
                            className="p-2 rounded-lg glass hover:bg-white/5 transition-colors"
                        >
                            <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
                        </button>
                    </div>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    <MetricCard
                        title="總收益 (PnL)"
                        value={`$${mockMetrics.totalPnL.toLocaleString()}`}
                        change={mockMetrics.pnlChange}
                        icon={<DollarSign className="w-5 h-5" />}
                        trend={mockMetrics.pnlChange >= 0 ? 'up' : 'down'}
                    />
                    <MetricCard
                        title="Sharpe Ratio"
                        value={mockMetrics.sharpeRatio.toFixed(2)}
                        change={mockMetrics.sharpeChange}
                        icon={<Target className="w-5 h-5" />}
                        trend={mockMetrics.sharpeChange >= 0 ? 'up' : 'down'}
                    />
                    <MetricCard
                        title="Win Rate"
                        value={`${mockMetrics.winRate}%`}
                        change={mockMetrics.winRateChange}
                        icon={<TrendingUp className="w-5 h-5" />}
                        trend={mockMetrics.winRateChange >= 0 ? 'up' : 'down'}
                    />
                    <MetricCard
                        title="總交易次數"
                        value={mockMetrics.totalTrades.toLocaleString()}
                        subtitle="今日新增: 58"
                        icon={<Activity className="w-5 h-5" />}
                    />
                </div>

                {/* Charts Section */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                    <div className="lg:col-span-2 glass rounded-2xl p-6">
                        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <BarChart3 className="w-5 h-5 text-indigo-400" />
                            收益趨勢
                        </h2>
                        <PerformanceChart data={mockChartData} />
                    </div>

                    <div className="glass rounded-2xl p-6">
                        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Zap className="w-5 h-5 text-amber-400" />
                            快速狀態
                        </h2>
                        <div className="space-y-4">
                            <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                                <span className="text-gray-400">活躍模型</span>
                                <span className="font-semibold text-emerald-400">{mockMetrics.activeModels}</span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                                <span className="text-gray-400">最大回撤</span>
                                <span className="font-semibold text-red-400">{mockMetrics.maxDrawdown}%</span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                                <span className="text-gray-400">系統狀態</span>
                                <span className="flex items-center gap-2">
                                    <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
                                    <span className="text-emerald-400">運行中</span>
                                </span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Models and Training */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <ModelList />
                    <TrainingStatus />
                </div>
            </main>
        </div>
    );
}
