import torch
import torch.nn as nn
import torch.nn.functional as F
from pymatgen.core import Composition
from tqdm import tqdm
from math import gcd
from functools import reduce

def reduce_formula(formula_str):
    """
    Reduce chemical formula to simplest form.
    
    Examples:
        Fe2O4 → Fe2O4 (already reduced)
        Fe4O8 → Fe2O4 (divide by 2)
        Fe6O12 → Fe2O4 (divide by 3)
        
    Args:
        formula_str: Original formula string
        
    Returns:
        reduced_formula: Reduced formula string
    """
    try:
        comp = Composition(formula_str)
        
        element_amounts = {}
        for element, amount in comp.items():
            element_amounts[element] = amount
        
        amounts = list(element_amounts.values())
        
        try:
            int_amounts = []
            for amt in amounts:
                # Handle floating point precision
                if abs(amt - round(amt)) < 1e-6:
                    int_amounts.append(int(round(amt)))
                else:
                    return comp.reduced_formula
            
            # Calculate GCD of all coefficients
            if len(int_amounts) > 0:
                common_gcd = reduce(gcd, int_amounts)
                
                if common_gcd > 1:
                    reduced_comp = {}
                    for element, amount in element_amounts.items():
                        reduced_comp[element] = int(round(amount)) // common_gcd
                    
                    # Rebuild formula string
                    sorted_elements = sorted(reduced_comp.keys(), key=lambda e: e.symbol)
                    reduced_formula = ""
                    for element in sorted_elements:
                        count = reduced_comp[element]
                        if count == 1:
                            reduced_formula += element.symbol
                        else:
                            reduced_formula += f"{element.symbol}{count}"
                    
                    return reduced_formula
                else:
                    return comp.reduced_formula
            else:
                return comp.reduced_formula
                
        except Exception as e:
            print(f"Reduction failed '{formula_str}': {e}, using default reduced_formula")
            return comp.reduced_formula
            
    except Exception as e:
        print(f"Formula parsing failed '{formula_str}': {e}")
        return formula_str


class FormulaLSTMEncoder(nn.Module):
    """LSTM-based chemical formula sequence encoder (preserves original order)"""
    def __init__(self, elem_embedding_dim=32, lstm_hidden_dim=64, 
                 num_lstm_layers=2, dropout=0.2):
        super().__init__()
        
        # Element embedding (1-100, 0 for padding)
        self.elem_embedding = nn.Embedding(101, elem_embedding_dim, padding_idx=0)
        
        # Atomic fraction encoder
        self.frac_encoder = nn.Linear(1, elem_embedding_dim // 2)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=elem_embedding_dim + elem_embedding_dim // 2,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_dim * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.output_dim = lstm_hidden_dim * 2
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, elem_indices, elem_fractions, seq_lengths):
        max_len = elem_indices.size(1)
        
        # Element embedding
        elem_emb = self.elem_embedding(elem_indices)
        
        # Atomic fraction encoding
        frac_emb = F.relu(self.frac_encoder(elem_fractions))
        
        # Concatenate embeddings
        combined_emb = torch.cat([elem_emb, frac_emb], dim=-1)
        
        # Pack sequence (handle variable length)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            combined_emb, seq_lengths.cpu(), 
            batch_first=True, enforce_sorted=False
        )
        
        # LSTM encoding
        packed_output, _ = self.lstm(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=max_len
        )
        
        # Self-attention
        attn_output, _ = self.attention(
            lstm_output, lstm_output, lstm_output
        )
        
        # Masked pooling
        mask = torch.arange(max_len, device=elem_indices.device)[None, :] < seq_lengths[:, None]
        mask = mask.unsqueeze(-1).float()
        
        masked_attn = attn_output * mask
        formula_features = masked_attn.sum(dim=1) / (seq_lengths.unsqueeze(-1).float() + 1e-8)
        
        return formula_features


class HybridFormulaEncoder(nn.Module):
    """Hybrid formula encoder: Bag(160-dim) + LSTM + fusion"""
    def __init__(self, bag_dim=160, lstm_hidden_dim=64, output_dim=128, dropout=0.2):
        super().__init__()
        
        # Bag encoder (160-dim → 64-dim)
        self.bag_encoder = nn.Sequential(
            nn.Linear(bag_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        # LSTM sequence encoder
        self.lstm_encoder = FormulaLSTMEncoder(
            elem_embedding_dim=32,
            lstm_hidden_dim=lstm_hidden_dim,
            num_lstm_layers=2,
            dropout=dropout
        )
        
        # Multi-modal fusion
        fusion_dim = 64 + lstm_hidden_dim * 2  # 64 + 128 = 192
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )
        
    def forward(self, bag_features, elem_indices, elem_fractions, seq_lengths):
        # Bag encoding
        bag_encoded = self.bag_encoder(bag_features)  # (batch, 64)
        
        # LSTM encoding
        lstm_encoded = self.lstm_encoder(elem_indices, elem_fractions, seq_lengths)  # (batch, 128)
        
        # Concatenate
        combined = torch.cat([bag_encoded, lstm_encoded], dim=-1)  # (batch, 192)
        
        # Self-attention enhancement
        combined_unsqueezed = combined.unsqueeze(1)
        attn_output, _ = self.cross_attention(
            combined_unsqueezed, combined_unsqueezed, combined_unsqueezed
        )
        attn_output = attn_output.squeeze(1)
        
        # Residual connection
        enhanced_combined = combined + attn_output
        
        # Final fusion
        fused = self.fusion(enhanced_combined)
        
        return fused


def formula_to_sequence_preserve_order(formula_str, max_len=10):
    """
    Convert formula to sequence preserving original order.
    
    Args:
        formula_str: "BaTiO3"
        max_len: Maximum sequence length
    
    Returns:
        elem_indices: [56(Ba), 22(Ti), 8(O), 0, ...]  (preserves Ba-Ti-O order)
        elem_fractions: [0.2, 0.2, 0.6, 0, ...]
        seq_length: 3
    """
    try:
        # Reduce formula
        formula_str = reduce_formula(formula_str)

        comp = Composition(formula_str)
        
        elem_indices = []
        elem_fractions = []
        
        # Use original element order (no sorting)
        for element in comp.elements:
            elem_indices.append(element.Z)
            elem_fractions.append(comp.get_atomic_fraction(element))
        
        # Padding
        seq_length = len(elem_indices)
        if seq_length < max_len:
            elem_indices += [0] * (max_len - seq_length)
            elem_fractions += [0.0] * (max_len - seq_length)
        else:
            elem_indices = elem_indices[:max_len]
            elem_fractions = elem_fractions[:max_len]
            seq_length = max_len
        
        return elem_indices, elem_fractions, seq_length
    
    except Exception as e:
        print(f"Formula sequencing failed '{formula_str}': {e}")
        return [0] * max_len, [0.0] * max_len, 0


def batch_preprocess_formulas_hybrid_160(formulas, bag_features_160, max_len=10):
    """
    Batch process formulas (hybrid mode: Bag-160 + sequence).
    
    Args:
        formulas: List of formulas
        bag_features_160: Pre-extracted 160-dim Bag features (tensor)
        max_len: Maximum sequence length
    
    Returns:
        bag_features_160: (N, 160) tensor
        seq_data: tuple of (elem_indices, elem_fractions, seq_lengths)
    """
    elem_indices_list = []
    elem_fractions_list = []
    seq_lengths_list = []
    
    print(f"Processing {len(formulas)} formula sequences (preserving order)...")
    
    for formula in tqdm(formulas, desc="Formula sequencing"):
        indices, fractions, length = formula_to_sequence_preserve_order(formula, max_len)
        elem_indices_list.append(indices)
        elem_fractions_list.append(fractions)
        seq_lengths_list.append(length)
    
    elem_indices = torch.tensor(elem_indices_list, dtype=torch.long)
    elem_fractions = torch.tensor(elem_fractions_list, dtype=torch.float32).unsqueeze(-1)
    seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.long)
    
    print(f"Sequencing complete (order preserved):")
    print(f"   Element indices: {elem_indices.shape}")
    print(f"   Atomic fractions: {elem_fractions.shape}")
    print(f"   Sequence length: avg {seq_lengths.float().mean():.2f}, range [{seq_lengths.min()}, {seq_lengths.max()}]")
    
    return bag_features_160, (elem_indices, elem_fractions, seq_lengths)


# Test function
if __name__ == "__main__":
    print("="*70)
    print("Testing order-preserving LSTM formula encoder")
    print("="*70)
    
    test_formulas = [
        "BaTiO3",         # Perovskite (large-small-small order)
        "TiO2",           # Rutile (small-small order)
        "Fe2O3",          # Hematite
        "La0.7Sr0.3MnO3", # Doped perovskite
    ]
    
    print("\nCompare sorted vs preserved order:")
    for formula in test_formulas:
        # Preserved order
        indices_preserve, fracs_preserve, _ = formula_to_sequence_preserve_order(formula, 10)
        
        # Sorted order (previous method)
        comp = Composition(formula)
        sorted_elements = sorted(comp.elements, key=lambda e: e.Z)
        indices_sorted = [e.Z for e in sorted_elements]
        fracs_sorted = [comp.get_atomic_fraction(e) for e in sorted_elements]
        
        print(f"\n{formula}:")
        print(f"  Preserved: elements={indices_preserve[:len(sorted_elements)]}, fractions={[f'{f:.3f}' for f in fracs_preserve[:len(sorted_elements)]]}")
        print(f"  Sorted: elements={indices_sorted}, fractions={[f'{f:.3f}' for f in fracs_sorted]}")
        print(f"  Difference: {'Order differs!' if indices_preserve[:len(sorted_elements)] != indices_sorted else 'Same order'}")
    
    print("\nTest complete!")
