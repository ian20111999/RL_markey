'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    LayoutDashboard,
    BrainCircuit,
    LineChart,
    Settings,
    Activity,
    Database,
    Newspaper,
    LogOut,
} from 'lucide-react';

const navItems = [
    { href: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { href: '/models', icon: BrainCircuit, label: '模型管理' },
    { href: '/training', icon: Activity, label: '訓練監控' },
    { href: '/backtest', icon: LineChart, label: '回測分析' },
    { href: '/data', icon: Database, label: '資料管理' },
    { href: '/logs', icon: Newspaper, label: '系統日誌' },
    { href: '/settings', icon: Settings, label: '設定' },
];

export default function Sidebar() {
    const pathname = usePathname();

    return (
        <aside className="fixed left-0 top-0 h-full w-64 glass border-r border-white/5 flex flex-col z-50">
            {/* Logo */}
            <div className="p-6 border-b border-white/5">
                <Link href="/" className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl gradient-bg flex items-center justify-center">
                        <BrainCircuit className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="font-bold text-lg">RL Market</h1>
                        <p className="text-xs text-gray-500">Trading System</p>
                    </div>
                </Link>
            </div>

            {/* Navigation */}
            <nav className="flex-1 p-4 space-y-1">
                {navItems.map((item) => {
                    const isActive = pathname === item.href;
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={`
                flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200
                ${isActive
                                    ? 'bg-gradient-to-r from-indigo-500/20 to-purple-500/20 text-white border border-indigo-500/30'
                                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                                }
              `}
                        >
                            <item.icon className={`w-5 h-5 ${isActive ? 'text-indigo-400' : ''}`} />
                            <span className="font-medium">{item.label}</span>
                        </Link>
                    );
                })}
            </nav>

            {/* User section */}
            <div className="p-4 border-t border-white/5">
                <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-white/5">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 flex items-center justify-center text-sm font-bold">
                        U
                    </div>
                    <div className="flex-1">
                        <p className="text-sm font-medium">User</p>
                        <p className="text-xs text-gray-500">Admin</p>
                    </div>
                    <button className="p-1.5 rounded-lg hover:bg-white/10 transition-colors">
                        <LogOut className="w-4 h-4 text-gray-400" />
                    </button>
                </div>
            </div>
        </aside>
    );
}
