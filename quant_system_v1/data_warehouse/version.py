"""Data snapshot version management — track data freshness and allow rollback."""
import os
import json
import pandas as pd
from datetime import datetime


class DataVersion:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.version_file = os.path.join(data_dir, 'data_version.json')

    def snapshot(self, label: str = None):
        """Record current data state with timestamp and file hashes."""
        import hashlib
        snapshot = {'timestamp': datetime.now().isoformat(), 'label': label, 'files': {}}
        if os.path.isdir(self.data_dir):
            for f in sorted(os.listdir(self.data_dir)):
                fp = os.path.join(self.data_dir, f)
                if os.path.isfile(fp) and f.endswith('.csv'):
                    with open(fp, 'rb') as fh:
                        snapshot['files'][f] = {
                            'size': os.path.getsize(fp),
                            'sha256': hashlib.sha256(fh.read()).hexdigest(),
                        }
        history = self._load_history()
        history.append(snapshot)
        with open(self.version_file, 'w') as f:
            json.dump(history, f, indent=2)
        return snapshot

    def latest(self) -> dict:
        """Return most recent snapshot."""
        history = self._load_history()
        return history[-1] if history else {}

    def list_versions(self) -> list:
        """List all snapshots."""
        return self._load_history()

    def check_freshness(self, max_age_hours: int = 24) -> dict:
        """Check if data is within the freshness window."""
        latest = self.latest()
        if not latest:
            return {'fresh': False, 'reason': 'no snapshot', 'age_hours': None}
        ts = datetime.fromisoformat(latest['timestamp'])
        age = (datetime.now() - ts).total_seconds() / 3600
        return {'fresh': age <= max_age_hours, 'age_hours': age,
                'last_update': latest['timestamp'], 'label': latest.get('label')}

    def _load_history(self) -> list:
        if os.path.exists(self.version_file):
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return []
