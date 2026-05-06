"""Model version management and periodic retraining."""
import os, sys, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from .predictor import MLPredictor

logger = get_logger("model_mgr")


class ModelManager:
    def __init__(self, model_dir=None, max_versions=3):
        base = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = model_dir or os.path.join(base, '..', 'evolution_output', 'models')
        self.max_versions = max_versions
        self.predictor = MLPredictor(self.model_dir)
        self.manifest_file = os.path.join(self.model_dir, 'manifest.json')

    def retrain(self, df, force=False):
        X, y = self.predictor.prepare_data(df)
        auc = self.predictor.train(X, y)
        if auc is None:
            return False
        current_auc = self._current_best_auc()
        if force or current_auc is None or auc > current_auc:
            version = pd.Timestamp.now().strftime('%Y%m%d')
            self.predictor.save(version)
            self._update_manifest(version, auc)
            self._cleanup_old()
            logger.info(f"New model {version}: AUC={auc:.4f} (prev={current_auc})")
            return True
        logger.info(f"No improvement: {auc:.4f} <= {current_auc:.4f}")
        return False

    def _current_best_auc(self):
        manifest = self._load_manifest()
        return max(v['auc'] for v in manifest.values()) if manifest else None

    def _update_manifest(self, version, auc):
        manifest = self._load_manifest()
        manifest[version] = {'auc': auc, 'date': pd.Timestamp.now().isoformat()}
        with open(self.manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

    def _cleanup_old(self):
        manifest = self._load_manifest()
        sorted_v = sorted(manifest.items(), key=lambda x: x[1]['auc'], reverse=True)
        for version, _ in sorted_v[self.max_versions:]:
            path = os.path.join(self.model_dir, f'model_{version}.pkl')
            if os.path.exists(path):
                os.remove(path)
            manifest.pop(version, None)
        with open(self.manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

    def _load_manifest(self):
        if os.path.exists(self.manifest_file):
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        return {}
