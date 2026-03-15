import torch
import numpy as np
import json


from .model_network import (
    ImprovedXRDNetWithFormula,
    batch_preprocess_formulas_enhanced_v2
)

# prepare the data and model for prediction

class XRDPredictor:
    """Simple XRD prediction interface"""

    def __init__(self, checkpoint_path, mapping_json, device='cpu'):
        """
        Args:
            checkpoint_path: Path to model weights (.pth)
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load class mappings from JSON
        with open(mapping_json, 'r') as f:
            mapping_data = json.load(f)
            self.label_to_idx = {k: int(v) for k, v in mapping_data['label_to_idx'].items()}
            self.idx_to_label = {int(k): v for k, v in mapping_data['idx_to_label'].items()}
            self.num_classes = mapping_data['num_classes']

        # Get model config
        self.formula_dim = checkpoint.get('formula_dim', 160)
        self.target_dim = checkpoint.get('target_dim', 3500)
        self.max_seq_len = checkpoint.get('max_seq_len', 10)
        self.dropout = checkpoint.get('dropout', 0.3)

        # Build model
        self.model = ImprovedXRDNetWithFormula(
            input_dim=self.target_dim,
            num_classes=self.num_classes,
            dropout_rate=self.dropout,
            formula_dim=self.formula_dim,
            use_formula=True
        ).to(self.device)

        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print(f"\nModel loaded")
        print(f"Device: {self.device}")
        print(f"Classes: {self.num_classes}\n")



    def normalize_xrd(self, xrd_data):
        """
        Robust minmax normalization for XRD data

        Args:
            xrd_data: Raw XRD intensity

        Returns:
            Normalized array
        """
        #xrd_data = np.array([float(x.strip()) for x in xrd_data.split(',')], dtype=np.float32)
        xrd_data = np.array(xrd_data, dtype=np.float32)
        # Ensure xrd_data is 1D
        if xrd_data.ndim > 1:
            xrd_data = xrd_data.flatten()
        # Compute quartiles and IQR
        q25 = np.percentile(xrd_data, 25)
        q75 = np.percentile(xrd_data, 75)
        iqr = q75 - q25
        if iqr == 0:
            iqr = 1.0
        # Clip outliers
        data_clipped = np.clip(xrd_data, q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        # Min-max normalization to [0, 1]
        data_min = np.min(data_clipped)
        data_max = np.max(data_clipped)
        data_range = data_max - data_min
        if data_range == 0:
            data_range = 1.0
        normalized = (data_clipped - data_min) / data_range
        return normalized

    def prepare_xrd_tensor(self, xrd_data):
        """
        Normalize and pad/crop XRD to target dimension

        Args:
            xrd_data: XRD intensity values

        Returns:
            Tensor of shape (1, target_dim)
        """
        # Normalize
        xrd_normalized = self.normalize_xrd(xrd_data)

        # Pad or crop
        current_dim = len(xrd_normalized)
        if current_dim < self.target_dim:
            padding = np.zeros(self.target_dim - current_dim)
            xrd_normalized = np.concatenate([xrd_normalized, padding])
        elif current_dim > self.target_dim:
            xrd_normalized = xrd_normalized[:self.target_dim]

        xrd_tensor = torch.FloatTensor(xrd_normalized).unsqueeze(0).to(self.device)
        return xrd_tensor

    def prepare_formula_tensor(self, formula):
        """
        Prepare formula tensor with hybrid encoding (Bag + LSTM)

        Args:
            formula: Chemical formula string (e.g. 'Fe2O3')

        Returns:
            Tuple of (bag_features, (elem_indices, elem_fractions, seq_lengths))
        """
        # Bag features
        bag_features = batch_preprocess_formulas_enhanced_v2([formula])
        # print(bag_features.shape)

        # # LSTM sequence features
        # comp = Composition(formula)
        # elements = list(comp.elements)[:self.max_seq_len]

        # elem_idx = [e.Z for e in elements]
        # elem_frac = [comp.get_atomic_fraction(e) for e in elements]
        # seq_len = len(elements)

        # # Padding
        # while len(elem_idx) < self.max_seq_len:
        #     elem_idx.append(0)
        #     elem_frac.append(0.0)

        # elem_indices = torch.LongTensor([elem_idx]).to(self.device)
        # elem_fractions = torch.FloatTensor([elem_frac]).unsqueeze(-1).to(self.device)
        # seq_lengths = torch.LongTensor([seq_len]).to(self.device)

        bag_features = bag_features.to(self.device)


        return bag_features

    def predict(self, xrd_data, formula, top_k=10, use_normalization=False):
        """
        Predict space group

        Args:
            xrd_data: XRD intensity (array or list)
            formula: Chemical formula (required)
            top_k: Number of top predictions to return

        Returns:
            List of (space_group, probability) tuples
        """
        # Prepare inputs
        if use_normalization:
            xrd_tensor = self.prepare_xrd_tensor(xrd_data)
        else:
            xrd_tensor = torch.FloatTensor(xrd_data).unsqueeze(0).to(self.device)
        formula_data = self.prepare_formula_tensor(formula)

        # Inference
        with torch.no_grad():
            outputs = self.model(xrd_tensor, formula_data)
            probabilities = torch.softmax(outputs, dim=1)[0]
            top_probs, top_indices = torch.topk(probabilities, k=min(top_k, self.num_classes))

        # Format results
        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            space_group = self.idx_to_label[int(idx)]
            predictions.append((space_group, float(prob)))

        return predictions


def load_model(checkpoint_path, mapping_json, device='cpu'):
    """
    Load pretrained model

    Args:
        checkpoint_path: Path to model weights
        device: 'cpu' or 'cuda'

    Returns:
        XRDPredictor instance
    """
    return XRDPredictor(checkpoint_path, mapping_json, device=device)



def predict_from_array(model, xrd_data, formula, top_k=10, use_normalization = False):
    """
    Predict from array data

    Args:
        model: XRDPredictor instance
        xrd_data: XRD intensity values
        formula: Chemical formula (required)
        top_k: Number of predictions

    Returns:
        List of (space_group, probability) tuples
    """
    return model.predict(xrd_data, formula, top_k, use_normalization = use_normalization)


