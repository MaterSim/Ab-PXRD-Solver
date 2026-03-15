# peak_finder/predict.py
"""Inference script - Find peaks using sliding window"""
import torch
import numpy as np
from tqdm import tqdm

from config import Config
from model import PeakFinderCNN
from data_utils import parse_csv_line


def sliding_window_predict(model, xrd, window_size=51, stride=1, device='cpu'):
    """
    Predict peak probabilities for the whole XRD using a sliding window.
    Returns:
        predictions: probability of peak at each position
    """
    model.eval()
    half_window = window_size // 2
    predictions = np.zeros(len(xrd))
    
    with torch.no_grad():
        for center in range(half_window, len(xrd) - half_window, stride):
            start = center - half_window
            end = start + window_size
            window = xrd[start:end]
            
            window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
            output = model(window_tensor)
            prob = torch.softmax(output, dim=1)[0, 1].item()  # probability of peak
            
            predictions[center] = prob
    
    return predictions


def find_peaks_from_predictions(predictions, threshold=0.5, min_distance=1):
    """Find peaks from predicted probabilities"""
    peaks = []
    for i in range(len(predictions)):
        if predictions[i] > threshold:
            # Check if local maximum
            is_peak = True
            for j in range(max(0, i-min_distance), min(len(predictions), i+min_distance+1)):
                if j != i and predictions[j] >= predictions[i]:
                    is_peak = False
                    break
            if is_peak:
                peaks.append(i)
    return peaks


def evaluate(model, xrd, true_labels, cfg, device):
    """Evaluate one XRD sample"""
    predictions = sliding_window_predict(
        model, xrd, 
        window_size=cfg.WINDOW_SIZE,
        stride=cfg.STRIDE,
        device=device
    )
    
    predicted_peaks = find_peaks_from_predictions(
        predictions,
        threshold=cfg.THRESHOLD
    )
    
    true_labels = np.atleast_1d(true_labels)
    true_peaks = np.where(true_labels == 1)[0].tolist()
    
    tp = len(set(predicted_peaks) & set(true_peaks))
    fp = len(set(predicted_peaks) - set(true_peaks))
    fn = len(set(true_peaks) - set(predicted_peaks))
    
    precision = tp / len(true_peaks) if len(true_peaks) > 0 else 0
    recall = tp / len(predicted_peaks) if len(predicted_peaks) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        
    return {
        'predictions': predictions,
        'predicted_peaks': predicted_peaks,
        'true_peaks': true_peaks,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def visualize_prediction(xrd, true_peaks, predicted_peaks, predictions, sample_id, save_dir='./visualization'):
    """Visualize XRD with true and predicted peaks"""
    import os
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 6))
    
    # Plot XRD pattern
    plt.subplot(2, 1, 1)
    plt.plot(xrd, 'k-', linewidth=0.8, label='XRD Pattern')
    
    # Mark true peaks
    if len(true_peaks) > 0:
        plt.scatter(true_peaks, xrd[true_peaks], c='green', s=100, 
                    marker='o', label=f'True Peaks ({len(true_peaks)})', zorder=5)
    
    # Mark predicted peaks
    if len(predicted_peaks) > 0:
        plt.scatter(predicted_peaks, xrd[predicted_peaks], c='red', s=50, 
                    marker='x', label=f'Predicted Peaks ({len(predicted_peaks)})', zorder=5)
    
    # Mark TP, FP, FN
    tp_peaks = list(set(predicted_peaks) & set(true_peaks))
    fp_peaks = list(set(predicted_peaks) - set(true_peaks))
    fn_peaks = list(set(true_peaks) - set(predicted_peaks))
    
    if len(tp_peaks) > 0:
        plt.scatter(tp_peaks, xrd[tp_peaks], c='blue', s=150, 
                    marker='*', label=f'TP ({len(tp_peaks)})', zorder=6)
    
    plt.xlabel('2θ Position')
    plt.ylabel('Intensity')
    plt.title(f'Sample {sample_id}: Peak Detection Results')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Plot prediction probabilities
    plt.subplot(2, 1, 2)
    plt.plot(predictions, 'b-', linewidth=0.8, label='Peak Probability')
    plt.axhline(y=0.85, color='r', linestyle='--', linewidth=0.8, label='Threshold')
    
    # Mark true peak positions
    if len(true_peaks) > 0:
        for peak in true_peaks:
            plt.axvline(x=peak, color='green', alpha=0.3, linewidth=1)
    
    plt.xlabel('2θ Position')
    plt.ylabel('Probability')
    plt.title('Peak Detection Probability')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sample_{sample_id}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"  TP: {len(tp_peaks)}, FP: {len(fp_peaks)}, FN: {len(fn_peaks)}")
    
    # Calculate metrics
    precision = len(tp_peaks) / len(predicted_peaks) if len(predicted_peaks) > 0 else 0
    recall = len(tp_peaks) / len(true_peaks) if len(true_peaks) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'predictions': predictions,
        'predicted_peaks': predicted_peaks,
        'true_peaks': true_peaks,
        'tp_peaks': tp_peaks,
        'fp_peaks': fp_peaks,
        'fn_peaks': fn_peaks,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == '__main__':
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = PeakFinderCNN(window_size=cfg.WINDOW_SIZE).to(device)
    checkpoint = torch.load(f'{cfg.SAVE_DIR}/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Read test data
    with open(cfg.testing_data_file, 'r', encoding='utf-8') as f:
        lines = []
        for i, line in enumerate(f):
            if i >= 30:
                break
            lines.append(line)
        
    parsed_data = []
    failed_count = 0
    
    # Parse each line
    for line in lines:
        result = parse_csv_line(line)
        if result is not None:
            parsed_data.append(result)
        else:
            failed_count += 1
    
    print(f"Parsed {len(parsed_data)} lines, failed {failed_count} lines.")
    if not parsed_data:
        raise Exception("No data parsed successfully!")
    
    ids = [item[0] for item in parsed_data]
    formulas = [item[1].replace(" ", "") for item in parsed_data]
    print(f"Formulas: {formulas[:5]}")
    labels = [item[4] for item in parsed_data]
    profiles = [item[3] for item in parsed_data]
    idx1 = 0
    precisions = []
    recalls = []
    f1s = []
    # Evaluate each sample
    for xrd, label in tqdm(zip(profiles, labels), total=len(profiles), desc="Evaluating"):
        xrd = (xrd - np.min(xrd)) / (np.max(xrd) - np.min(xrd) + 1e-8)
        result = evaluate(model, xrd, label, cfg, device)
        idx1 += 1
        print(f"\nSample {idx1}:")
        print(f"  Precision: {result['precision']:.3f}")
        print(f"  Recall: {result['recall']:.3f}")
        print(f"  F1: {result['f1']:.3f}")
        print(f"  True peaks: {len(result['true_peaks'])}")
        print(f"  Predicted peaks: {len(result['predicted_peaks'])}")
        precisions.append(result['precision'])
        recalls.append(result['recall'])
        
        # 添加可视化调用
        visualize_prediction(
            xrd=xrd,
            true_peaks=result['true_peaks'],
            predicted_peaks=result['predicted_peaks'],
            predictions=result['predictions'],
            sample_id=idx1,
            save_dir='./visualization'
        )
    if precisions:
        print("\n=== Average Metrics on Test Set ===")
        print(f"Average Precision: {np.mean(precisions):.3f}")
        print(f"Average Recall: {np.mean(recalls):.3f}")
        print(f"Average F1: {np.mean(f1s):.3f}")

        
        
    # for xrd, label in zip(profiles, labels):
    #     # xrd = parse_array_from_string(profile_str)
    #     # label = parse_array_from_string(label_str).astype(np.int32)

    #     # normalize xrd
    #     xrd = (xrd - np.min(xrd)) / (np.max(xrd) - np.min(xrd) + 1e-8)
    #     result = evaluate(model, xrd, labels, cfg, device)
    #     idx+=1
    #     print(f"\nSample {idx}:")
    #     print(f"  Precision: {result['precision']:.3f}")
    #     print(f"  Recall: {result['recall']:.3f}")
    #     print(f"  F1: {result['f1']:.3f}")
    #     print(f"  True peaks: {len(result['true_peaks'])}")
    #     print(f"  Predicted peaks: {len(result['predicted_peaks'])}")

    # print("开始预测...")
    # for idx, row in df.iterrows():
    #     xrd = parse_array_string(row['profile'])
    #     xrd = (xrd - xrd.min()) / (xrd.max() - xrd.min() + 1e-8)
    #     labels = parse_array_string(row['label']).astype(int)
        
    #     result = evaluate(model, xrd, labels, cfg, device)
        
    #     print(f"\nSample {idx}:")
    #     print(f"  Precision: {result['precision']:.3f}")
    #     print(f"  Recall: {result['recall']:.3f}")
    #     print(f"  F1: {result['f1']:.3f}")
    #     print(f"  True peaks: {len(result['true_peaks'])}")
    #     print(f"  Predicted peaks: {len(result['predicted_peaks'])}")