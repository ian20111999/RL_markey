# Frontend 前端目錄

本目錄包含所有前端相關的檔案，包括 Web 儀表板、HTML 模板等。

## 📁 目錄結構

```
frontend/
├── dashboards/          # 儀表板 HTML 檔案
│   ├── basic.html       # 基礎儀表板
│   ├── enhanced.html    # 增強版儀表板
│   └── production.html  # 生產級儀表板（來自 production/templates/）
├── static/              # 靜態資源（CSS, JS, 圖片等）
└── README.md           # 本文件
```

## 🎯 儀表板說明

### 1. basic.html（基礎儀表板）
- **用途**: 簡單的訓練進度監控
- **啟動**: `python web_dashboard.py`
- **端口**: 5555
- **特點**: 輕量級，基礎功能

### 2. enhanced.html（增強版儀表板）
- **用途**: 進階監控與數據可視化
- **啟動**: `python monitoring_dashboard.py`
- **端口**: 5556
- **特點**: 圖表豐富，即時更新

### 3. production.html（生產級儀表板）
- **用途**: 企業級模型管理與監控
- **啟動**: `python production/dashboard.py`
- **端口**: 8080
- **特點**: 完整的模型註冊系統整合

## 🚀 使用方式

### 快速啟動基礎儀表板
```bash
python web_dashboard.py
# 訪問 http://localhost:5555
```

### 啟動增強版儀表板
```bash
python monitoring_dashboard.py
# 訪問 http://localhost:5556
```

### 啟動生產級儀表板
```bash
python production/dashboard.py
# 訪問 http://localhost:8080
```

## 📝 開發指南

### 添加新的儀表板

1. 在 `dashboards/` 目錄創建新的 HTML 檔案
2. 如果需要靜態資源（CSS/JS），放在 `static/` 目錄
3. 更新本 README 說明使用方式

### 靜態資源管理

```python
# 在 Flask app 中配置靜態目錄
app = Flask(__name__, 
            template_folder='frontend/dashboards',
            static_folder='frontend/static')
```

## 🔗 相關文檔

- [Web Dashboard 使用指南](../docs/USER_GUIDE_ZH.md#web-監控面板)
- [生產級部署指南](../docs/PRODUCTION_GUIDE.md)
- [API 文檔](../production/api.py)
