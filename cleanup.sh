#!/bin/bash
# 清理 a-stock-quant-v3 数据缓存中超过30天的旧文件

CACHE_DIR="/root/.openclaw/workspace/a-stock-quant-v3/data_cache"
LOG_FILE="/root/.openclaw/workspace/a-stock-quant-v3/cleanup.log"

if [ ! -d "$CACHE_DIR" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') 目录不存在: $CACHE_DIR" >> "$LOG_FILE"
    exit 1
fi

# 删除30天前的CSV缓存文件（保留akshare_factors_和最新的选股结果）
DELETED=$(find "$CACHE_DIR" -type f -name "*.csv" -mtime +30 -not -name "akshare_factors_*.csv" -not -name "stock_picks_*.csv" -print | wc -l)
find "$CACHE_DIR" -type f -name "*.csv" -mtime +30 -not -name "akshare_factors_*.csv" -not -name "stock_picks_*.csv" -delete

# 清理其他旧文件
DELETED_OTHERS=$(find "$CACHE_DIR" -type f -mtime +30 -not -name "*.csv" -print | wc -l)
find "$CACHE_DIR" -type f -mtime +30 -not -name "*.csv" -delete

AFTER_SIZE=$(du -sh "$CACHE_DIR" | cut -f1)
AFTER_COUNT=$(find "$CACHE_DIR" -type f | wc -l)

echo "$(date '+%Y-%m-%d %H:%M:%S') 清理完成 | 删除CSV: $DELETED, 其他: $DELETED_OTHERS | 剩余: $AFTER_COUNT files, $AFTER_SIZE" >> "$LOG_FILE"
