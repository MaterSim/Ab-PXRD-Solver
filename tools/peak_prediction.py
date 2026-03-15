import sys
from pathlib import Path

# Add peak_finder to path
peak_path = Path(__file__).parent / "peak_finder"
if peak_path.exists():
    sys.path.insert(0, str(peak_path.parent))

# Add spacegroup to path
spg_path = Path(__file__).parent / "spacegroup"
if spg_path.exists():
    sys.path.insert(0, str(spg_path.parent))

import torch
import numpy as np
from peak_finder.config import Config
from peak_finder.model import PeakFinderCNN
from spacegroup.spacegroup_predictor import load_model, predict_from_array

def sliding_window_predict(model, xrd, window_size=51, device='cpu'):
    """
    Predict peak probabilities for the whole XRD using a sliding window.

    Args:
        model: trained peak prediction model
        xrd: 1D numpy array of XRD intensities
        window_size: size of the sliding window
        device: computation device ('cpu' or 'cuda')

    Returns:
        predictions: probability of peak at each position
    """
    model.eval()
    half_window = window_size // 2
    # each size pad half_window at both ends
    background =  np.percentile(xrd, 10)
    padded_xrd = np.pad(xrd, (half_window, half_window), mode='constant',
                        constant_values=background)
    # padded_xrd.shape -> (3500 + window_size -1, )
    predictions = np.zeros(len(xrd))

    with torch.no_grad():
        for center in range(len(xrd)):
            start = center
            end = start + window_size
            window = padded_xrd[start:end]

            window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
            output = model(window_tensor)
            prob = torch.softmax(output, dim=1)[0, 1].item()  # probability of peak

            predictions[center] = prob

    return predictions

def predict_spacegroup(xrd, formula, top_k=10, use_normalization=False):
    """
    Predict space groups from the given intensity array using the provided model.
    """
    chkpt = Path(__file__).parent / 'models/spacegroup/best_model.pth'
    mapping_json = Path(__file__).parent / 'models/spacegroup/label_mapping.json'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(chkpt, mapping_json, device=device)

    predictions = predict_from_array(model, xrd, formula, top_k=top_k, use_normalization = use_normalization)
    return predictions

def _predict_peaks(xrd):
    """
    Predict peaks in the given intensity array using the provided model.
    """
    current_dir = Path(__file__).parent
    chkpt = str(current_dir / "models" / "peaks" / "best_model.pth")
    # the new peak_finder checkpoint can be found at https://github.com/qzhu2017/PXRD-GPT/tree/strands-multiagent/get_stat_3500/spacegroup_peak_scale_mask_GSAS_withcleandata_nonoise_20260205_023156

    # Load model
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PeakFinderCNN(window_size=cfg.WINDOW_SIZE).to(device)
    checkpoint = torch.load(chkpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    predictions = sliding_window_predict(
        model, xrd,
        window_size=cfg.WINDOW_SIZE,
        device=device
    )
    return predictions

def predict_peaks(xrd, threshold=0.5, min_height=5.0):
    """
    Predict peaks in the given intensity array using the provided model.
    """
    predictions = _predict_peaks(xrd)
    mask1 = predictions > threshold
    mask2 = xrd > min_height
    mask = mask1 | mask2

    candidates = np.where(mask)[0]

    # Remove consecutive IDs, keeping only the one with maximum xrd value
    filtered_candidates = []
    i = 0
    while i < len(candidates):
        # Find the end of consecutive sequence
        j = i
        while j + 1 < len(candidates) and candidates[j + 1] == candidates[j] + 1:
            j += 1

        # candidates[i:j+1] are consecutive
        if j > i:
            # Multiple consecutive IDs - find the one with max xrd value
            consecutive_ids = candidates[i:j+1]
            max_idx = consecutive_ids[np.argmax(xrd[consecutive_ids])]
            filtered_candidates.append((max_idx, predictions[max_idx]))
        else:
            # Single ID
            filtered_candidates.append((candidates[i], predictions[candidates[i]]))

        i = j + 1

    return filtered_candidates

def _infer_formula_spg(path: Path):
    """Infer formula and space group from a file name like PXRD_<formula>_<spg>.csv."""
    tokens = path.stem.split('_')
    formula_guess, spg_guess = None, None
    if len(tokens) >= 2:
        try:
            spg_guess = int(tokens[-1])
            # Join middle tokens to support names with extra underscores.
            formula_guess = '_'.join(tokens[1:-1]) if len(tokens) > 2 else None
        except ValueError:
            pass
    return formula_guess, spg_guess

if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    from XRD import Profile

    pxrd_csvs = [#'Examples/PXRD_Ba14Na14LiN6_225.csv',
                 #'Examples/PXRD_Ba4NaBi_216.csv',
                 #'Examples/PXRD_DyB6_221.csv',
                 #'Examples/PXRD_PbS_186.csv',
                 #'Examples/PXRD_Mg9Si5_176.csv',
                 #'Examples/PXRD_PrYMg2_123.csv',
                 #'Examples/PXRD_TbMnSi_62.csv',
                 'Examples/PXRD_TiCuSiAs_129.csv',
                 #'Examples/PXRD_Be2SiBi_119.csv'
                 ]

    for pxrd_csv in pxrd_csvs:
        formula, spg = _infer_formula_spg(Path(pxrd_csv))
        df = pd.read_csv(pxrd_csv, comment='#')
        x1 = df.iloc[:, 0].values
        y1 = df.iloc[:, 1].values
        # y1 = y1 / np.max(y1) #+ 1e-8  # normalize
        y1 = (y1 - np.min(y1)) / (np.max(y1) - np.min(y1) + 1e-8)   # use min-max normalization：: (x - min) / (max - min) instead of normalize
        results = predict_peaks(y1, 0.8)
        print(results)
        # Prepare peak positions and intensities
        peak_positions = [pos for pos, prob in results]
        peak_intensities = [y1[pos]*100 for pos in peak_positions]

        # Build reconstructed profile
        (px, py) = Profile("gaussian").get_profile(x1[peak_positions], peak_intensities, 10, 80)
        print(px, py)

        print("\n=== Testing two data formats ===")
        #predictions_y1 = predict_spacegroup(y1, formula, top_k=10, use_normalization=True)
        #print(f"y1 (original) predictions: {predictions_y1}")

        predictions_py = predict_spacegroup(py, formula, top_k=10, use_normalization=False)
        print(f"py (reconstructed profile) predictions: {predictions_py}")

        # Create 2x1 subplot: top = original XRD with predicted peaks, bottom = reconstructed profile
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Top subplot: original XRD + predicted peaks
        axes[0].plot(x1, y1*100, label='XRD')
        axes[0].scatter(x1[peak_positions], peak_intensities, color='red',
                        label='Predicted Peaks', alpha=0.4)
        for pos, prob in results:
            axes[0].text(x1[pos], y1[pos]*100, f"{prob:.2f}", color='red',
                         fontsize=8, verticalalignment='bottom', horizontalalignment='center')
        axes[0].legend()
        axes[0].set_ylabel('Normalized Intensity')
        axes[0].set_title('XRD with Predicted Peaks')

        # Bottom subplot: reconstructed XRD profile
        axes[1].plot(px, py, label='Reconstructed XRD Profile', color='green', linestyle='--')
        axes[1].legend()
        axes[1].set_xlabel('2θ (degrees)')
        axes[1].set_ylabel('Intensity')
        axes[1].set_title('Reconstructed XRD Profile')


        strs_py = f"\nReconstructed Profile (py):\n"
        for space_group, probability in predictions_py:
            strs_py += f"  {space_group}: {probability:.2%}\n"

        strs_full = f"{pxrd_csv}\nFormula: {formula}\n{strs_py}"

        # Add space group predictions to the bottom subplot
        axes[1].text(0.65, 0.98, strs_full.strip(), transform=axes[1].transAxes,
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                     fontsize=9)

        fig.tight_layout()
        fig.savefig(f'{formula}.png', dpi=300)

