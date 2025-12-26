#!/bin/bash
# 等待 PostgreSQL 資料庫準備就緒

set -e

host="${POSTGRES_HOST:-localhost}"
port="${POSTGRES_PORT:-5432}"
user="${POSTGRES_USER:-rl_user}"
db="${POSTGRES_DB:-rl_market}"

echo "等待 PostgreSQL 在 $host:$port 準備就緒..."

max_attempts=30
attempt=0

until PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$host" -p "$port" -U "$user" -d "$db" -c '\q' 2>/dev/null; do
  attempt=$((attempt + 1))
  
  if [ $attempt -ge $max_attempts ]; then
    echo "錯誤: PostgreSQL 在 $max_attempts 次嘗試後仍未準備就緒"
    exit 1
  fi
  
  echo "PostgreSQL 尚未準備就緒 - 等待中 (嘗試 $attempt/$max_attempts)..."
  sleep 2
done

echo "PostgreSQL 已準備就緒！"

# 執行傳入的命令
exec "$@"
