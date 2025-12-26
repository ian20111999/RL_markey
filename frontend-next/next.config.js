/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  
  // API 代理到後端
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ];
  },
  
  // 環境變數
  env: {
    API_BASE_URL: process.env.API_BASE_URL || 'http://localhost:8000',
  },
};

module.exports = nextConfig;
