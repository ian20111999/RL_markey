import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
    title: 'RL Market Dashboard',
    description: 'Reinforcement Learning Trading System Dashboard',
    icons: {
        icon: '/favicon.ico',
    },
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="zh-TW" className="dark">
            <body className="min-h-screen bg-[#0a0a0f] antialiased">
                {children}
            </body>
        </html>
    );
}
