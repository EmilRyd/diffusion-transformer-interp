import torch
import numpy as np
from torch.utils.data import Dataset

class ShiftDataset(Dataset):
    def __init__(self, image_size=4, for_cnn=False):
        self.rows, self.cols = image_size, image_size
        self.num_samples = (self.rows * (self.cols - 1))  # 4 * 3 = 12 for 4x4
        self.for_cnn = for_cnn  # Flag to toggle format
        self.valid_positions = [(r, c) for r in range(self.rows) for c in range(self.cols - 1)]
        
        inputs = []
        outputs = []
        # Systematically add all valid positions
        for row, col in self.valid_positions:
            input_grid = np.zeros((self.rows, self.cols), dtype=np.float32)
            input_grid[row, col] = 1.0  # Place white pixel
            output_grid = np.zeros((self.rows, self.cols), dtype=np.float32)
            output_grid[:, 1:] = input_grid[:, :-1]  # Shift right
            if self.for_cnn:
                inputs.append(input_grid)  # Keep as 4x4 for CNN
                outputs.append(output_grid)
            else:
                inputs.append(input_grid.flatten())  # Flatten for transformer
                outputs.append(output_grid.flatten())
        
        self.inputs = torch.tensor(inputs)
        self.outputs = torch.tensor(outputs)
        if self.for_cnn:
            self.inputs = self.inputs.unsqueeze(1)  # (N, 1, 4, 4)
            self.outputs = self.outputs.unsqueeze(1)  # (N, 1, 4, 4)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    


class ContinuousShiftDataset(Dataset):
    def __init__(self, num_samples=10, image_size=3, for_cnn=False):
        self.rows, self.cols = image_size, image_size
        self.num_samples = num_samples
        self.for_cnn = for_cnn  # Toggle between flattened (transformer) or 2D (CNN) format
        
        # Generate random input images and their shifted outputs
        inputs = []
        outputs = []
        for _ in range(num_samples):
            # Random 3x3 image with values between 0 and 1
            input_grid = np.random.uniform(low=0.0, high=1.0, size=(self.rows, self.cols)).astype(np.float32)
            
            # Shift right with wrap-around (rightmost column moves to leftmost)
            output_grid = np.zeros_like(input_grid)
            output_grid[:, 1:] = input_grid[:, :-1]  # Shift all columns right
            output_grid[:, 0] = input_grid[:, -1]   # Wrap rightmost column to left
            
            if self.for_cnn:
                inputs.append(input_grid)  # Keep as 3x3 for CNN
                outputs.append(output_grid)
            else:
                inputs.append(input_grid.flatten())  # Flatten to 9 for transformer
                outputs.append(output_grid.flatten())
        
        self.inputs = torch.tensor(inputs)
        self.outputs = torch.tensor(outputs)
        if self.for_cnn:
            self.inputs = self.inputs.unsqueeze(1)   # (N, 1, 3, 3)
            self.outputs = self.outputs.unsqueeze(1)  # (N, 1, 3, 3)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]