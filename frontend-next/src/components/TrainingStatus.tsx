'use client';

import { Activity, Play, Pause, Clock } from 'lucide-react';

const mockTrainingRuns = [
    {
        id: 'run_btc_001',
        symbol: 'BTCUSDT',
        algorithm: 'SAC',
        status: 'running',
        progress: 67,
        episode: 670,
        totalEpisodes: 1000,
        eta: '23 min',
        reward: 245.8,
    },
    {
        id: 'run_eth_002',
        symbol: 'ETHUSDT',
        algorithm: 'PPO',
        status: 'paused',
        progress: 45,
        episode: 450,
        totalEpisodes: 1000,
        eta: '38 min',
        reward: 178.2,
    },
];

export default function TrainingStatus() {
    return (
        <div className="glass rounded-2xl p-6">
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold flex items-center gap-2">
                    <Activity className="w-5 h-5 text-emerald-400" />
                    訓練狀態
                </h2>
                <a href="/training" className="text-sm text-indigo-400 hover:underline">
                    管理 →
                </a>
            </div>

            <div className="space-y-4">
                {mockTrainingRuns.map((run) => (
                    <div key={run.id} className="p-4 rounded-xl bg-white/5">
                        <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-3">
                                <div className={`w-2 h-2 rounded-full ${run.status === 'running' ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
                                <span className="font-medium">{run.symbol}</span>
                                <span className="text-xs text-gray-500 px-2 py-0.5 rounded bg-white/10">
                                    {run.algorithm}
                                </span>
                            </div>
                            <button className="p-1.5 rounded-lg hover:bg-white/10 transition-colors">
                                {run.status === 'running' ? (
                                    <Pause className="w-4 h-4 text-gray-400" />
                                ) : (
                                    <Play className="w-4 h-4 text-gray-400" />
                                )}
                            </button>
                        </div>

                        {/* Progress bar */}
                        <div className="relative h-2 bg-white/10 rounded-full mb-3 overflow-hidden">
                            <div
                                className="absolute h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-500"
                                style={{ width: `${run.progress}%` }}
                            >
                                {run.status === 'running' && (
                                    <div className="absolute right-0 top-0 bottom-0 w-4 bg-gradient-to-l from-white/30 to-transparent animate-pulse" />
                                )}
                            </div>
                        </div>

                        <div className="flex items-center justify-between text-sm">
                            <span className="text-gray-400">
                                Episode {run.episode}/{run.totalEpisodes}
                            </span>
                            <div className="flex items-center gap-4">
                                <span className="text-emerald-400">
                                    Reward: {run.reward.toFixed(1)}
                                </span>
                                <span className="flex items-center gap-1 text-gray-500">
                                    <Clock className="w-3 h-3" />
                                    {run.eta}
                                </span>
                            </div>
                        </div>
                    </div>
                ))}

                {mockTrainingRuns.length === 0 && (
                    <div className="text-center py-8 text-gray-500">
                        <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
                        <p>目前沒有進行中的訓練</p>
                    </div>
                )}
            </div>
        </div>
    );
}
