"""Continuous learning scheduler: daily/weekly/monthly automated tasks."""
import os, sys, json
import pandas as pd
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.calendar import TradeCalendar

logger = get_logger("scheduler")


class EvolutionScheduler:
    def __init__(self):
        self.cal = TradeCalendar()
        self.last_run = self._load_state()

    def run_daily(self):
        today = datetime.now().strftime('%Y-%m-%d')
        if today == self.last_run.get('daily'):
            return
        logger.info(f"[DAILY] {today}")
        self.last_run['daily'] = today
        self._save_state()

    def run_weekly(self):
        week = datetime.now().strftime('%Y-W%W')
        if week == self.last_run.get('weekly'):
            return
        logger.info(f"[WEEKLY] {week}: Factor lifecycle check")
        from evolution.factor_evolution import FactorLifecycleMonitor
        monitor = FactorLifecycleMonitor()
        from factor_lib.registry import FactorRegistry
        for name in FactorRegistry.list_all():
            status = monitor.check(name)
            if status['action'] in ('REMOVE', 'WARN'):
                logger.warning(f"  {name}: {status['status']} IC={status.get('avg_ic',0):.4f}")
        self.last_run['weekly'] = week
        self._save_state()

    def run_monthly(self):
        month = datetime.now().strftime('%Y-%m')
        if month == self.last_run.get('monthly'):
            return
        logger.info(f"[MONTHLY] {month}: Evolution cycle")
        self.last_run['monthly'] = month
        self._save_state()

    def run_all(self):
        self.run_daily()
        self.run_weekly()
        self.run_monthly()

    def _load_state(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, 'evolution_output', 'scheduler_state.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def _save_state(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, 'evolution_output', 'scheduler_state.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.last_run, f, indent=2)
