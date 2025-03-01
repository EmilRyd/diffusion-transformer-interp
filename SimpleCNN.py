import torch.nn as nn
# 2. Define the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)  # (batch, 1, IMAGE_SIZE, IMAGE_SIZE) -> (batch, 1, 2, 2) without padding
        #x = nn.functional.pad(x, (1, 1, 1, 1))  # Back to (batch, 1, IMAGE_SIZE, IMAGE_SIZE)
        x = self.sigmoid(x)
        return x
