"""
Predict density range for a composition.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add aviary to path
aviary_path = Path(__file__).parent / "aviary"
if aviary_path.exists():
    sys.path.insert(0, str(aviary_path.parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from aviary.roost.data import CompositionData, collate_batch
from aviary.roost.model import Roost
import glob

class DensityEnsemblePredictor:
    """Load models once and reuse for efficient batch predictions."""

    def __init__(self, checkpoint_pattern: Optional[str] = None):
        """Initialize by loading all models."""
        if checkpoint_pattern is None:
            current_dir = Path(__file__).parent
            checkpoint_pattern = str(current_dir / "models" / "density" / "checkpoint-r*.pth.tar")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_paths = sorted(glob.glob(checkpoint_pattern))
        if not checkpoint_paths:
            raise FileNotFoundError(f"No density model checkpoints found matching: {checkpoint_pattern}")

        self.models = []
        self.scales = []
        self.means = []

        # Load all models once
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model = Roost(**checkpoint["model_params"])
            model.to(self.device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()

            self.models.append(model)
            normalizer = checkpoint.get("normalizer_dict", {}).get("density", None)
            if normalizer:
                self.scales.append(float(normalizer["std"]))
                self.means.append(float(normalizer["mean"]))
            else:
                self.scales.append(1.0)
                self.means.append(0.0)

    def predict(self, composition: str, sigma: float = 3.0) -> Dict[str, Any]:
        """Predict density for a single composition using pre-loaded models."""
        predictions = []
        uncertainties = []

        df = pd.DataFrame({'material_id': ['temp'], 'composition': [composition], 'density': [0]})
        dataset = CompositionData(df=df, task_dict={'density': 'regression'})
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_batch)

        with torch.no_grad():
            for inputs, *_ in loader:
                inputs = [inp.to(self.device) for inp in inputs]
                for model, scale, mean in zip(self.models, self.scales, self.means):
                    output = model(*inputs)[0].cpu()
                    prediction, log_std = output[0]

                    prediction = prediction.item() * scale + mean
                    uncertainty = torch.exp(log_std).item() * scale

                    predictions.append(prediction)
                    uncertainties.append(uncertainty)

        # Aggregate ensemble results
        mean_pred = sum(predictions) / len(predictions)
        mean_uncertainty = sum(uncertainties) / len(uncertainties)
        return {
            'composition': composition,
            'prediction': mean_pred,
            'uncertainty': mean_uncertainty,
            'min': mean_pred - sigma * mean_uncertainty,
            'max': mean_pred + sigma * mean_uncertainty,
        }


def predict_density_ensemble(composition: str,
                             sigma: float = 3.0,
                             checkpoint_pattern: Optional[str] = None,
                             predictor: Optional[DensityEnsemblePredictor] = None) -> Dict[str, Any]:
    """Backward compatible function. Use DensityEnsemblePredictor for better performance."""
    if predictor is None:
        predictor = DensityEnsemblePredictor(checkpoint_pattern)
    return predictor.predict(composition, sigma)

if __name__ == "__main__":
    from ase.db import connect

    # Load models once
    predictor = DensityEnsemblePredictor()

    db = connect("total.db")
    failures = 0

    for row in db.select():
        formula = row.formula
        density = row.get("density", None)
        r = predict_density_ensemble(formula, sigma=3, predictor=predictor)
        if r['min'] < density < r['max']:
            print(f"i:{row.id} {formula:20s} ([{r['min']:6.3f}, {r['max']:6.3f}] g/cm³) covers {density:6.3f}")
        else:
            print(f"i:{row.id} {formula:20s} ([{r['min']:6.3f}, {r['max']:6.3f}] g/cm³) does not cover {density:6.3f}")
            failures += 1
    print(f"\nTotal failures: {failures} out of {db.count()}")
