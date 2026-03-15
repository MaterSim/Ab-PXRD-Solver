import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import re
def parse_csv_line(line):
    try:
        line = line.strip()
        if not line or line.startswith('id'):
            return None
        
        quote_pattern = r'"([^"]*)"'
        matches = re.findall(quote_pattern, line)
        
        if not matches:
            return None
        
        profile_str = matches[0]
        label_str = matches[1] if len(matches) > 1 else ''
        line_without_profile = re.sub(quote_pattern, 'PROFILE_PLACEHOLDER', line, count=1)
        fields = line_without_profile.split(',')
        
        if len(fields) < 3:
            return None
        
        jid = fields[0].strip()
        formula = fields[1].strip()
        spacegroup = int(fields[2].strip())
        profile_values = [float(x.strip()) for x in profile_str.split(',') if x.strip()]
        label_values =[int(x.strip()) for x in label_str.split(',') if x.strip()]
        
        return jid, formula, spacegroup, profile_values, label_values
        
    except Exception:
        return None


class PeakDataset(Dataset):
    def __init__(self, csv_file, window_size=51, samples_per_xrd=10, positive_ratio=0.4):
        """
        Custom dataset class for loading XRD data and corresponding peak labels.

        Args:
            csv_file (str): Path to the CSV file containing XRD data and peak labels.
        """
        print("loading data from:", csv_file)
        self.df = pd.read_csv(csv_file,dtype={'profile': str, 'label': str})
        self.window_size = window_size  
        self.samples_per_xrd = samples_per_xrd
        self.positive_ratio = positive_ratio
        self.half_window = window_size // 2

        # Analyze XRD and label data
        print("analysing xrd and label data...")
        self.xrd_list = []
        self.label_list = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        parsed_data = []
        failed_count = 0
        
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

        for xrd, label in zip(profiles, labels):
            # Normalize XRD
            xrd = (xrd - np.min(xrd)) / (np.max(xrd) - np.min(xrd) + 1e-8)
            
            self.xrd_list.append(xrd)
            self.label_list.append(label)
        print("data analysis complete.", len(self.xrd_list), "XRD patterns loaded.")
    
    def __len__(self):
        return len(self.xrd_list) * self.samples_per_xrd
    
    def __getitem__(self, idx):
        # Determine which XRD pattern and which sample within that pattern
        xrd_idx = idx // self.samples_per_xrd
        sample_idx = idx % self.samples_per_xrd

        xrd = self.xrd_list[xrd_idx]
        label = self.label_list[xrd_idx]

        # Find the range that can be used for sampling
        valid_positions = np.arange(self.half_window, len(xrd) - self.half_window)

        # Determine number of positive and negative samples
        positive_pos = [p for p in valid_positions if label[p] == 1]
        negative_pos = [p for p in valid_positions if label[p] == 0]
        
        # Randomly select a position
        if np.random.rand() < self.positive_ratio and len(positive_pos) > 0:
            # Select positive sample
            center = np.random.choice(positive_pos)
            y = 1
        else:
            # Select negative sample
            if len(negative_pos) == 0:
                center = np.random.choice(valid_positions)
            else:
                center = np.random.choice(negative_pos)
            y = label[center]

        # Get window
        start = center - self.half_window
        end = center + self.half_window + 1
        x_window = xrd[start:end]

        return {
            'window': torch.FloatTensor(x_window),
            'label': torch.LongTensor([y])[0],
            'position': center
        }
