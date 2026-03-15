# peak_finder/predict.py
"""Inference script - Find peaks using sliding window"""
import torch
import numpy as np
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from peak_finder.config import Config
from peak_finder.model import PeakFinderCNN
from peak_finder.data_utils import parse_csv_line
import json
# from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
# from jarvis.analysis.diffraction.xrd import XRD as JXRD
from pyxtal import pyxtal
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import os
import numpy as np
import matplotlib.pyplot as plt
from math import gcd
from scipy.signal import find_peaks
from pyxtal.XRD import Similarity
from pyxtal.database.element import Element
from scipy.signal import find_peaks
from ase.db import connect
from pyxtal.XRD import Similarity
from pyxtal.interface.ase_opt import ASE_relax
from pyxtal.database.element import Element
from tools.manager import RawDataManager, CellManager
from tools.gsas import simulate_pxrd
from tools.XRD import XRD


def sliding_window_predict(model, xrd, window_size=51, stride=1, device='cpu'):
    """
    Predict peak probabilities for the whole XRD using a sliding window.
    Returns:
        predictions: probability of peak at each position
    """
    model.eval()
    half_window = window_size // 2
    # each size pad half_window at both ends
    background =  np.percentile(xrd, 10)
    padded_xrd = np.pad(xrd, (half_window, half_window), mode='constant', constant_values=background)
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


def find_peaks_from_predictions(predictions, threshold=0.5, min_distance=1, true_peak = None):
    """Find peaks from predicted probabilities"""
    
    candidates = []
    for i in range(len(predictions)):
        if predictions[i] > threshold:
            candidates.append((i, predictions[i]))  # (index, probability)

            # Check if local maximum
            #is_peak = True
            # for j in range(max(0, i-min_distance), min(len(predictions), i+min_distance+1)):
            #     if j != i an
            # d predictions[j] >= predictions[i]:
            #         is_peak = False
            #         break
            

    #直接返回所有候选峰的位置
    peaks = [idx for idx, prob in candidates]
    return peaks
    
    # sort by probability   from high to low
    # candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    # min_distance = 1
    # # non-maximum suppression
    # peaks = []
    # suppressed = set()  # record suppressed indices
    # for pos, prob in candidates:
    #     if pos in suppressed:
    #         continue
    #     for other_pos, _ in candidates:
    #         if abs(other_pos - pos) <= min_distance and other_pos != pos:
    #             if true_peak[pos]!=1 and true_peak[other_pos] == 1:
    #                 suppressed.add(pos)
    #                 continue
    #             suppressed.add(other_pos)

    # #
    # peaks = [pos for pos, _ in candidates if pos not in suppressed]
    # peaks = sorted(peaks)

    # return peaks


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
        threshold=cfg.THRESHOLD,
        min_distance=1,
        true_peak=true_labels
    )
    
    true_labels = np.atleast_1d(true_labels)
    true_peaks = np.where(true_labels == 1)[0].tolist()
    
    tp = len(set(predicted_peaks) & set(true_peaks))
    fp = len(set(predicted_peaks) - set(true_peaks))
    fn = len(set(true_peaks) - set(predicted_peaks))
    
    recall = tp / (len(true_peaks)/2) if len(true_peaks) > 0 else 0
    precision = tp / len(predicted_peaks) if len(predicted_peaks) > 0 else 0
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
    """Visualize XRD with true and predicted peaks and save peak data to txt files"""
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建更大的画布，3行1列的子图
    fig = plt.figure(figsize=(20, 12))
    original_predicted_peaks = predicted_peaks
    
    # Mark TP, FP, FN
    tp_peaks = list(set(predicted_peaks) & set(true_peaks))
    fp_peaks = list(set(predicted_peaks) - set(true_peaks))
    fn_peaks = list(set(true_peaks) - set(predicted_peaks))
    print(f"Sample {sample_id} - TP: {len(tp_peaks)}, FP: {len(fp_peaks)}, FN: {len(fn_peaks)}")

    # true_peaks is always even-length, each peak is a pair, take the first of each pair
    half_true_peak = [true_peaks[i] for i in range(0, len(true_peaks), 2)]


    filltered_predicted_peaks = []
    for peak in tp_peaks:
        higher_than_neighbor = True
        for tp in tp_peaks:
            if abs(tp - peak) == 1 and predictions[peak]<predictions[tp]:
                higher_than_neighbor = False
                break
        if higher_than_neighbor:
            filltered_predicted_peaks.append(peak)

    fillter_false_peaks = []
    for peak in fp_peaks:
        higher_than_neighbor = True
        for tp in tp_peaks:
            if abs(tp - peak) == 1 and predictions[peak]<predictions[tp]:
                higher_than_neighbor = False
                break
        if higher_than_neighbor:
            fillter_false_peaks.append(peak)
    
    # Expand filltered_predicted_peaks by adding left and right neighbors
    expanded_peaks = set()
    for peak in filltered_predicted_peaks:
        expanded_peaks.add(peak - 1)
        if(peak-2)>=0:
            expanded_peaks.add(peak - 2)
        expanded_peaks.add(peak)
        expanded_peaks.add(peak + 1)
        expanded_peaks.add(peak + 2)
    
    expanded_true_peaks = set()
    for peak in true_peaks:
        expanded_true_peaks.add(peak - 1)
        if(peak-2)>=0:
            expanded_true_peaks.add(peak - 2)
        expanded_true_peaks.add(peak)
        expanded_true_peaks.add(peak + 1)
        expanded_true_peaks.add(peak + 2)



    fillter_false_peaks = list(set(fillter_false_peaks)-(set(fillter_false_peaks)&set(expanded_true_peaks)))
    
    # Convert to sorted list and filter out invalid indices
    #filltered_predicted_peaks_expanded = sorted([p for p in expanded_peaks if 0 <= p < len(xrd)])

    predicted_peaks = filltered_predicted_peaks + fillter_false_peaks

    fn_peaks = list(set(true_peaks)- set(expanded_peaks))


    # filter peaks for better visualization
    #filtered_true_peaks = []
    #filtered_true_peaks = filltered_predicted_peaks + fn_peaks
    
    # ============ 子图1: XRD Pattern with All Peaks ============
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(xrd, 'k-', linewidth=0.8, label='XRD Pattern', alpha=0.6)
    
    # Mark TP (蓝色星星)
    # if len(tp_peaks) > 0:
    #     ax1.scatter(tp_peaks, xrd[tp_peaks], c='blue', s=150, 
    #                 marker='*', alpha=0.8, label=f'TP ({len(tp_peaks)})', zorder=6)
    #     # 标注 TP 索引
    #     for peak in tp_peaks:
    #         ax1.annotate(f'{peak}', xy=(peak, xrd[peak]), 
    #                     xytext=(0, 10), textcoords='offset points',
    #                     fontsize=7, ha='center', color='blue', weight='bold')
    
    # # Mark FP (红色叉)
    # if len(fp_peaks) > 0:
    #     ax1.scatter(fp_peaks, xrd[fp_peaks], c='red', s=100, 
    #                 marker='x', alpha=0.8, label=f'FP ({len(fp_peaks)})', zorder=5, linewidths=2)
    #     # 标注 FP 索引
    #     for peak in fp_peaks:
    #         ax1.annotate(f'{peak}', xy=(peak, xrd[peak]), 
    #                     xytext=(0, -15), textcoords='offset points',
    #                     fontsize=7, ha='center', color='red')
    
    # Mark FN (绿色圆圈)
    if len(half_true_peak) > 0:
        fn_filtered = [p for p in half_true_peak]
        if len(fn_filtered) > 0:

            ax1.scatter(fn_filtered, xrd[fn_filtered], c='green', s=100, 
                        marker='o', alpha=0.6, label=f'true peak ({len(half_true_peak)})', zorder=4)
            # 标注 FN 索引
            for peak in fn_filtered:
                ax1.annotate(f'{peak}', xy=(peak, xrd[peak]), 
                            xytext=(0, 10), textcoords='offset points',
                            fontsize=7, ha='center', color='green')
    
    ax1.set_xlabel('2θ Position', fontsize=10)
    ax1.set_ylabel('Intensity', fontsize=10)
    ax1.set_title(f'Sample {sample_id}: Peak Detection Results (All Peaks)', fontsize=12, weight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)


    # ============ 子图2: FN Peak ============
    ax1 = plt.subplot(4, 1, 2)
    ax1.plot(xrd, 'k-', linewidth=0.8, label='Fn peak', alpha=0.6)
    
    
    # Mark FN (绿色圆圈)
    if len(fn_peaks) > 0:
        fn_filtered = [p for p in fn_peaks]
        if len(fn_filtered) > 0:
            ax1.scatter(fn_filtered, xrd[fn_filtered], c='red', s=100, 
                        marker='o', alpha=0.6, label=f'Unpredicted peak ({len(fn_peaks)})', zorder=4)
            # 标注 FN 索引
            for peak in fn_filtered:
                ax1.annotate(f'{peak}', xy=(peak, xrd[peak]), 
                            xytext=(0, 10), textcoords='offset points',
                            fontsize=7, ha='center', color='red')
    
    ax1.set_xlabel('2θ Position', fontsize=10)
    ax1.set_ylabel('Intensity', fontsize=10)
    ax1.set_title(f'Sample {sample_id}: Unpredicted Peaks', fontsize=12, weight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    
    # ============ 子图3: 只显示 TP 和 FP (Predicted Peaks) ============
    ax2 = plt.subplot(4, 1, 3)
    ax2.plot(xrd, 'k-', linewidth=0.8, label='XRD Pattern', alpha=0.6)
    
    # TP
    if len(filltered_predicted_peaks) > 0:
        ax2.scatter(filltered_predicted_peaks, xrd[filltered_predicted_peaks], c='blue', s=150, 
                    marker='*', alpha=0.8, label=f'TP ({len(filltered_predicted_peaks)})', zorder=6)
        for peak in filltered_predicted_peaks:
            ax2.annotate(f'{peak}', xy=(peak, xrd[peak]), 
                        xytext=(0, 10), textcoords='offset points',
                        fontsize=8, ha='center', color='blue', weight='bold')
    
    # FP
    if len(fillter_false_peaks) > 0:
        ax2.scatter(fillter_false_peaks, xrd[fillter_false_peaks], c='red', s=100, 
                    marker='x', alpha=0.8, label=f'FP ({len(fillter_false_peaks)})', zorder=5, linewidths=2)
        for peak in fillter_false_peaks:
            ax2.annotate(f'{peak}', xy=(peak, xrd[peak]), 
                        xytext=(0, -15), textcoords='offset points',
                        fontsize=8, ha='center', color='red')
    
    ax2.set_xlabel('2θ Position', fontsize=10)
    ax2.set_ylabel('Intensity', fontsize=10)
    ax2.set_title('Predicted Peaks: TP (correct) vs FP (false positive)', fontsize=12, weight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ============ 子图4: Peak Probability ============
    ax3 = plt.subplot(4, 1, 4)
    ax3.plot(predictions, 'b-', linewidth=0.8, label='Peak Probability')
    ax3.axhline(y=0.8, color='r', linestyle='--', linewidth=1.5, label='Threshold', alpha=0.7)
    
    # 标记 true peak 位置（绿色虚线）
    if len(half_true_peak) > 0:
        for peak in half_true_peak:
            ax3.axvline(x=peak, color='green', alpha=0.3, linewidth=1, linestyle='dotted')
    
    # 标记 predicted peaks
    if len(predicted_peaks) > 0:
        ax3.scatter(predicted_peaks, [predictions[p] for p in predicted_peaks], 
                   c='red', s=30, marker='v', alpha=0.6, zorder=5)
    
    ax3.set_xlabel('2θ Position', fontsize=10)
    ax3.set_ylabel('Probability', fontsize=10)
    ax3.set_title('Peak Detection Probability', fontsize=12, weight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sample_{sample_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data files
    np.savetxt(f'{save_dir}/sample_{sample_id}_true_peaks.txt', sorted(true_peaks), fmt='%d', header='True Peaks (Indices)')
    np.savetxt(f'{save_dir}/sample_{sample_id}_predicted_peaks.txt', sorted(original_predicted_peaks), fmt='%d', header='Predicted Peaks (Indices)')
    np.savetxt(f'{save_dir}/sample_{sample_id}_predictions.txt', predictions, fmt='%.6f', header='Predictions (Probabilities)')
    
    # Print statistics
    print(f"  TP: {len(tp_peaks)}, FP: {len(fp_peaks)}, FN: {len(fn_peaks)}")
    
    # Calculate metrics
    precision = len(filltered_predicted_peaks) / len(predicted_peaks) if len(predicted_peaks) > 0 else 0
    recall = len(filltered_predicted_peaks) / (len(true_peaks)/2) if len(true_peaks) > 0 else 0
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

# def visualize_prediction(xrd, true_peaks, predicted_peaks, predictions, sample_id, save_dir='./visualization'):
#     """Visualize XRD with true and predicted peaks and save peak data to txt files"""
#     import os
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     os.makedirs(save_dir, exist_ok=True)
    
#     plt.figure(figsize=(15, 6))
    
#     # Plot XRD pattern
#     plt.subplot(2, 1, 1)
#     plt.plot(xrd, 'k-', linewidth=0.8, label='XRD Pattern')
    
#     # Mark true peaks
#     # if len(true_peaks) > 0:
#     #     plt.scatter(true_peaks, xrd[true_peaks], c='green', s=100, 
#     #                 marker='o', alpha=0.6, label=f'True Peaks ({len(true_peaks)})', zorder=5)
    
#     # Mark predicted peaks
#     if len(predicted_peaks) > 0:
#         plt.scatter(predicted_peaks, xrd[predicted_peaks], c='red', s=50, 
#                     marker='x', alpha=0.6, label=f'Predicted Peaks ({len(predicted_peaks)})', zorder=5)
    
#     # Mark TP, FP, FN
#     tp_peaks = list(set(predicted_peaks) & set(true_peaks))
#     fp_peaks = list(set(predicted_peaks) - set(true_peaks))
#     fn_peaks = list(set(true_peaks) - set(predicted_peaks))

#     # filter peaks for better visualization
#     filtered_true_peaks = []
#     for peak in true_peaks:
#         has_neighbor_in_tp = False
#         for tp in tp_peaks:
#             if abs(tp - peak) == 1:
#                 has_neighbor_in_tp = True
#                 break
#         if peak in tp_peaks or not has_neighbor_in_tp:
#             filtered_true_peaks.append(peak)

#     if len(tp_peaks) > 0:
#         plt.scatter(tp_peaks, xrd[tp_peaks], c='blue', s=150, 
#                     marker='*', alpha=0.6, label=f'TP ({len(tp_peaks)})', zorder=6)
        
#     #Mark true peaks
#     if len(filtered_true_peaks) > 0:
#         plt.scatter(filtered_true_peaks, xrd[filtered_true_peaks], c='green', s=100, 
#                     marker='o', alpha=0.6, label=f'True Peaks ({len(filtered_true_peaks)})', zorder=5)
    
#     plt.xlabel('2θ Position')
#     plt.ylabel('Intensity')
#     plt.title(f'Sample {sample_id}: Peak Detection Results')
#     plt.legend(loc='upper right')
#     plt.grid(True, alpha=0.3)
    
#     # Plot prediction probabilities
#     plt.subplot(2, 1, 2)
#     plt.plot(predictions, 'b-', linewidth=0.8, label='Peak Probability')
#     plt.axhline(y=0.98, color='r', linestyle='--', linewidth=0.8, label='Threshold')
    
#     # Mark true peak positions
#     if len(filtered_true_peaks) > 0:
#         for peak in filtered_true_peaks:
#             plt.axvline(x=peak, color='green', alpha=0.3, linewidth=1, linestyle='dotted')
    
#     plt.xlabel('2θ Position')
#     plt.ylabel('Probability')
#     plt.title('Peak Detection Probability')
#     plt.legend(loc='upper right')
#     plt.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(f'{save_dir}/sample_{sample_id}.png', dpi=150, bbox_inches='tight')
#     plt.close()
    
#     # Save true peaks and predicted peaks to txt files
#     np.savetxt(f'{save_dir}/sample_{sample_id}_true_peaks.txt', filtered_true_peaks, fmt='%d', header='True Peaks (Indices)')
#     np.savetxt(f'{save_dir}/sample_{sample_id}_predicted_peaks.txt', predicted_peaks, fmt='%d', header='Predicted Peaks (Indices)')
#     np.savetxt(f'{save_dir}/sample_{sample_id}_predictions.txt', predictions, fmt='%d', header='Predictions (Indices)')
    
#     # Print statistics
#     print(f"  TP: {len(tp_peaks)}, FP: {len(fp_peaks)}, FN: {len(fn_peaks)}")
    
#     # Calculate metrics
#     precision = len(tp_peaks) / len(predicted_peaks) if len(predicted_peaks) > 0 else 0
#     recall = len(tp_peaks) / (len(true_peaks)/2) if len(true_peaks) > 0 else 0
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

#     return {
#         'predictions': predictions,
#         'predicted_peaks': predicted_peaks,
#         'true_peaks': filtered_true_peaks,
#         'tp_peaks': tp_peaks,
#         'fp_peaks': fp_peaks,
#         'fn_peaks': fn_peaks,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1
#     }


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
    
    # Load XRD test data
    jsonfile = '/users/ksu4/atomgpt/PXRD-GPT/jdft_3d-8-18-2021.json'
    if os.path.exists(jsonfile):
        print("Loading existing JSON file...")
        with open(jsonfile, 'r') as f:
            dft_3d = json.load(f)
        print(f"Loaded {len(dft_3d)} entries from {jsonfile}")
    
        # target_ids = [
        #     'JVASP-91609', 'JVASP-98189', 'JVASP-18823', 'JVASP-88546', 'JVASP-92200', 'JVASP-97928',
        #     'JVASP-64606', 'JVASP-97505', 'JVASP-90341', 'JVASP-97684', 'JVASP-63924', 'JVASP-98097',
        #     'JVASP-5503', 'JVASP-64955', 'JVASP-22620', 'JVASP-65108', 'JVASP-14619', 'JVASP-85928',
        #     'JVASP-64332', 'JVASP-64610', 'JVASP-86579', 'JVASP-65066', 'JVASP-91849', 'JVASP-37208',
        #     'JVASP-48167', 'JVASP-88712'
        # ]
        target_ids = [ 'JVASP-91609']
        idx1 = 0
        precisions = []
        recalls = []
        f1s = []

        SCALED_INTENSITY_TOL=0.02
        wavelength = 1.54064
        resolution = 0.02
        thetas = [10, 80]
        pid = os.getpid()
        tmp_dir = f'tmp/worker_{pid}'
        os.makedirs(tmp_dir, exist_ok=True)

        for target_id in target_ids:
            entry = next((item for item in dft_3d if item['jid'] == target_id), None)
            ii = int(''.join(filter(str.isdigit, entry['jid'])))
            if entry is None:
                print(f"{target_id}: Not found in dataset.")
                continue
            at_dict = entry['atoms']
            pmg = Atoms.from_dict(at_dict).pymatgen_converter()
            xtal = pyxtal()
            xtal.from_seed(pmg)
            spg = xtal.group.number
            #xrd = xtal.get_XRD(thetas=[10, 80], SCALED_INTENSITY_TOL=0.02)
            #x1,y1 = xrd.get_plot(bg_ratio=0)
            #y1= np.array(y1)
            #profile = xrd.get_profile(res=0.02)

            origin_cif = f'Train/Structrue_{ii}.cif'
            #data_png = f'Train/Plot_{ii}.png'
            #peak_txt = f'Train/Peaks_{ii}.txt'

            Cell = xtal.lattice.encode()
            cell_str = ' '.join([f'{x:8.4f}' for x in Cell])
            title = f'{ii:4d} {spg:3d} {xtal.formula:16s} {cell_str}, {xtal.lattice.volume:7.1f}'
            xtal.to_file(origin_cif)
            #print(title)

            # Get XRD peaks
            x1, y1 = simulate_pxrd(origin_cif, iparams="tools/INST_XRY.PRM",
                                   Tmin=10, Tmax=80,
                                   U=0.2, V=-0.2, W=2.0, X=0.5, Y=0.5,
                                   bg_ratio=0, add_noise=True, noise_level=0.001, gpx_name=f'{tmp_dir}/simulation.gpx'
                                   )
            
            data = RawDataManager(x1, y1, bg_subtract=False)
            data.get_peaks_from_scipy_adaptive(prominence=5.0) # change me
            if len(data.peaks) < 2:
                print("Problem in generating xrd")
                continue
            

            xrd = XRD(xtal.to_ase(), wavelength=wavelength, thetas=thetas,
              res=resolution, SCALED_INTENSITY_TOL=SCALED_INTENSITY_TOL)
            # 获取理论峰位置 (不规则间距)
            theoretical_peaks_theta = xrd.pxrd[:, 0]  # 2theta values
            #np.savetxt(peak_txt, theoretical_peaks_theta)
            #print(theoretical_peaks_theta)
            #data.plot(data_png, title)
            
            

            peaks = []
            for peak_theta in theoretical_peaks_theta:
                # Find indices of the two closest points in x1 to peak_theta
                diffs = np.abs(x1 - peak_theta)
                closest_indices = np.argsort(diffs)[:2]
                peaks.extend(closest_indices)
            peaks = np.array(peaks)
            #print(peaks)

            #peaks, _ = find_peaks(y1, height=1, prominence=2.5)
            num_peaks = len(peaks)

            # create binary label array (3500 points)
            label = np.zeros(3500, dtype=int)
            for peak_idx in peaks:
                if peak_idx < 3500:
                    label[peak_idx] = 1
            
            xrd_np = y1

            formula = pmg.formula.replace(" ", "")
            xrd_np = (xrd_np - np.min(xrd_np)) / (np.max(xrd_np) - np.min(xrd_np) + 1e-8)
            results = evaluate(model, xrd_np, label, cfg, device)
            
            
            # 添加可视化调用
            result = visualize_prediction(
                xrd=xrd_np,
                true_peaks=results['true_peaks'],
                predicted_peaks=results['predicted_peaks'],
                predictions=results['predictions'],
                sample_id=idx1,
                save_dir='/scratch/ksu4/PXRD-GPT/peak_finder/visualization'
            )
            idx1 += 1
            print(f"\nSample {idx1}:")
            print(f"  Precision: {result['precision']:.3f}")
            print(f"  Recall: {result['recall']:.3f}")
            print(f"  F1: {result['f1']:.3f}")
            print(f"  True peaks: {len(result['true_peaks'])/2}")
            print(f"  Predicted peaks: {len(result['predicted_peaks'])}")
            precisions.append(result['precision'])
            recalls.append(result['recall'])
            f1s.append(result['f1'])
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