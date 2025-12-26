'use client';

import { BrainCircuit, Check, AlertTriangle, Clock } from 'lucide-react';

const mockModels = [
    {
        id: 'sac_btc_20251226',
        name: 'SAC-BTC-v3',
        symbol: 'BTCUSDT',
        algorithm: 'SAC',
        sharpe: 2.45,
        status: 'deployed',
        pnl: '+12.5%',
    },
    {
        id: 'ppo_eth_20251225',
        name: 'PPO-ETH-v2',
        symbol: 'ETHUSDT',
        algorithm: 'PPO',
        sharpe: 1.87,
        status: 'ready',
        pnl: '+8.3%',
    },
    {
        id: 'td3_sol_20251224',
        name: 'TD3-SOL-v1',
        symbol: 'SOLUSDT',
        algorithm: 'TD3',
        sharpe: 1.52,
        status: 'training',
        pnl: '+5.2%',
    },
];

const statusConfig = {
    deployed: {
        icon: Check,
        color: 'text-emerald-400',
        bg: 'bg-emerald-500/20',
        label: '已部署',
    },
    ready: {
        icon: Clock,
        color: 'text-amber-400',
        bg: 'bg-amber-500/20',
        label: '待部署',
    },
    training: {
        icon: AlertTriangle,
        color: 'text-blue-400',
        bg: 'bg-blue-500/20',
        label: '訓練中',
    },
};

export default function ModelList() {
    return (
        <div className="glass rounded-2xl p-6">
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold flex items-center gap-2">
                    <BrainCircuit className="w-5 h-5 text-purple-400" />
                    模型列表
                </h2>
                <a href="/models" className="text-sm text-indigo-400 hover:underline">
                    查看全部 →
                </a>
            </div>

            <div className="space-y-3">
                {mockModels.map((model) => {
                    const status = statusConfig[model.status as keyof typeof statusConfig];
                    const StatusIcon = status.icon;

                    return (
                        <div
                            key={model.id}
                            className="flex items-center justify-between p-4 rounded-xl bg-white/5 hover:bg-white/10 transition-colors cursor-pointer"
                        >
                            <div className="flex items-center gap-4">
                                <div className="w-10 h-10 rounded-lg gradient-bg flex items-center justify-center text-xs font-bold">
                                    {model.algorithm}
                                </div>
                                <div>
                                    <p className="font-medium">{model.name}</p>
                                    <p className="text-sm text-gray-500">{model.symbol}</p>
                                </div>
                            </div>

                            <div className="flex items-center gap-4">
                                <div className="text-right">
                                    <p className="text-sm text-emerald-400 font-medium">{model.pnl}</p>
                                    <p className="text-xs text-gray-500">Sharpe: {model.sharpe}</p>
                                </div>
                                <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full ${status.bg} ${status.color} text-xs`}>
                                    <StatusIcon className="w-3 h-3" />
                                    {status.label}
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
