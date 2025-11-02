"""
自动化监控脚本 - 每日定时运行
运行: python monitor.py
或设置crontab: 0 8 * * * python /path/to/monitor.py
"""

import sys
import os
from datetime import datetime
import schedule
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from pathlib import Path

# 导入主系统
from property_system import (
    RealDataFetcher,
    PropertyDatabase,
    TimeSeriesPredictor,
    AutomatedMonitor
)

# 配置
CONFIG_FILE = 'config.json'


class PropertyMonitorService:
    """房产监控服务"""

    def __init__(self, config_path: str = CONFIG_FILE):
        self.config = self.load_config(config_path)
        self.db = PropertyDatabase()