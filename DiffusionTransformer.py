import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from ShiftDataset import ShiftDataset, ContinuousShiftDataset

from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

# Diffusion Utilities (unchanged)
def linear_beta_schedule(timesteps, betas):
    return torch.linspace(*betas, timesteps)

class Diffusion:
    def __init__(self, timesteps=100, betas=[0.0001, 0.05]):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps, betas)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise):
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise

class DiffusionTransformer(nn.Module):
    def __init__(self, pixel_dim=16, embed_dim=4, num_heads=1, num_layers=1):
        super(DiffusionTransformer, self).__init__()
        self.pixel_dim = pixel_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Embeddings
        self.pixel_embed = nn.Linear(1, embed_dim)
        self.time_embed = nn.Linear(1, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, pixel_dim, embed_dim))
        
        # Transformer layers with multi-head attention
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'norm1': nn.LayerNorm(embed_dim),
                'ff': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.ReLU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                ),
                'norm2': nn.LayerNorm(embed_dim)
            }) for _ in range(num_layers)
        ])
        
        # Output layer
        self.out = nn.Linear(embed_dim, 1)
    
    def forward(self, x_noisy, t, x_cond, return_attention=False):
        batch_size = x_noisy.shape[0]
        
        # Embed noisy pixels
        x_noisy = x_noisy.unsqueeze(-1)
        x_embed = self.pixel_embed(x_noisy) + self.pos_embed
        
        # Embed timestep
        t = t.unsqueeze(-1)
        t_embed = self.time_embed(t)
        t_embed = t_embed.unsqueeze(1).expand(-1, self.pixel_dim, -1)
        x_embed = x_embed + t_embed
        
        # Embed conditioning input
        x_cond = x_cond.unsqueeze(-1)
        cond_embed = self.pixel_embed(x_cond) + self.pos_embed
        
        # Process through Transformer layers
        attn_weights = None
        for layer in self.layers:
            # Multi-head cross-attention (x_embed attends to cond_embed)
            attn_output, attn_weights = layer['attention'](x_embed, cond_embed, cond_embed)


            # get the att
            x_embed = layer['norm1'](x_embed + attn_output)  # Residual connection + normalization
            # Feedforward
            ff_output = layer['ff'](x_embed)
            x_embed = layer['norm2'](x_embed + ff_output)  # Residual connection + normalization

        # Predict noise
        noise_pred = self.out(x_embed).squeeze(-1)
        
        if return_attention:
            return noise_pred, attn_weights  # Returns weights from the last layer
        return noise_pred

    # Sampling Function (unchanged)
    '''@torch.no_grad()
    def sample(self, x_cond, timesteps=100):
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diffusion = Diffusion(timesteps=timesteps)
        x = torch.randn_like(x_cond)
        for t in reversed(range(timesteps)):
            t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.float)
            noise_pred = self(x, t_tensor, x_cond)
            alpha = diffusion.alphas[t]
            alpha_cumprod = diffusion.alphas_cumprod[t]
            beta = diffusion.betas[t]
            
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred)
            if t > 0:
                x += torch.sqrt(beta) * torch.randn_like(x)
        return torch.clamp(x, 0, 1)'''
    
    # Sampling Function (unchanged)
    @torch.no_grad()
    def sample(self, x_cond, timesteps=100, return_all=False):
        self.eval()

        x_list = []
        attn_weights_list = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diffusion = Diffusion(timesteps=timesteps)

        x = torch.randn_like(x_cond)

        for t in reversed(range(timesteps)):
            t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.float)
            noise_pred, attn = self(x, t_tensor, x_cond, return_attention=True)
            attn_weights_list.append(attn[0].cpu().numpy())
            alpha = diffusion.alphas[t]
            alpha_cumprod = diffusion.alphas_cumprod[t]
            beta = diffusion.betas[t]
            x_list.append(torch.clamp(x, 0, 1))
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred)
            if t > 0:
                x += torch.sqrt(beta) * torch.randn_like(x)
        if return_all:
            return torch.clamp(x, 0, 1), x_list, attn_weights_list
        else:
            return torch.clamp(x, 0, 1)
    

class SteerableDiffusionTransformer(DiffusionTransformer):
    def __init__(self, steering_vector, steering_alpha = 0.1, steering_point = 0, pixel_dim=16, embed_dim=4, num_heads=1, num_layers=1):
        super(SteerableDiffusionTransformer, self).__init__(pixel_dim, embed_dim, num_heads, num_layers)
        self.steering_vector = steering_vector
        self.steering_alpha = steering_alpha
        self.steering_point = steering_point

    def forward(self, x_noisy, t, x_cond, return_attention=False):
        # add self.steering_vector*steering_alpha to x_embed between every step

        batch_size = x_noisy.shape[0]
        
        # Embed noisy pixels
        x_noisy = x_noisy.unsqueeze(-1)
        x_embed = self.pixel_embed(x_noisy) + self.pos_embed
        
        # Embed timestep
        t = t.unsqueeze(-1)
        t_embed = self.time_embed(t)
        t_embed = t_embed.unsqueeze(1).expand(-1, self.pixel_dim, -1)
        x_embed = x_embed + t_embed
        
        # Embed conditioning input
        x_cond = x_cond.unsqueeze(-1)
        cond_embed = self.pixel_embed(x_cond) + self.pos_embed
        
        # Process through Transformer layers
        attn_weights = None
        for layer in self.layers:
            # Multi-head cross-attention (x_embed attends to cond_embed)
            attn_output, attn_weights = layer['attention'](x_embed, cond_embed, cond_embed)

            # possible white vector
            poss_white_vec = attn_output[0][np.where(x_cond[0] == 1.)[0][0]+1]


            # get the att
            x_embed = layer['norm1'](x_embed + attn_output)  # Residual connection + normalization

            # add steering_vector*steering_alpha to x_embed
            x_embed[0, self.steering_point] = x_embed[0, self.steering_point] + self.steering_vector*self.steering_alpha

            # Feedforward
            ff_output = layer['ff'](x_embed)
            x_embed = layer['norm2'](x_embed + ff_output)  # Residual connection + normalization

        # Predict noise
        noise_pred = self.out(x_embed).squeeze(-1)
        
        if return_attention:
            return noise_pred, attn_weights, poss_white_vec  # Returns weights from the last layer
        return noise_pred
    
    # Sampling Function (unchanged)
    @torch.no_grad()
    def sample(self, x_cond, timesteps=100, return_all=False):
        self.eval()

        x_list = []
        attn_weights_list = []
        poss_white_vecs = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diffusion = Diffusion(timesteps=timesteps)

        x = torch.randn_like(x_cond)

        for t in reversed(range(timesteps)):
            t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.float)
            noise_pred, attn, poss_white_vec = self(x, t_tensor, x_cond, return_attention=True)
            attn_weights_list.append(attn[0].cpu().numpy())
            poss_white_vecs.append(poss_white_vec.cpu().numpy())
            alpha = diffusion.alphas[t]
            alpha_cumprod = diffusion.alphas_cumprod[t]
            beta = diffusion.betas[t]
            x_list.append(torch.clamp(x, 0, 1))
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred)
            if t > 0:
                x += torch.sqrt(beta) * torch.randn_like(x)
        if return_all:
            return torch.clamp(x, 0, 1), x_list, attn_weights_list, poss_white_vecs
        else:
            return torch.clamp(x, 0, 1)
    



class DiffusionTrainer:
    # Train the diffusion transformer
    def __init__(self, image_size, embed_dim, learning_rate, gamma_, step_size, num_epochs, batch_size,num_layers, num_heads, timesteps, dataset=ShiftDataset, num_samples=1000):
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.gamma_ = gamma_
        self.num_epochs = num_epochs
        self.timesteps = timesteps
        self.batch_size = batch_size
        if dataset == ShiftDataset:
            self.dataset = dataset(image_size=self.image_size, for_cnn=False)
        elif dataset == ContinuousShiftDataset:
            self.dataset = dataset(num_samples=num_samples, image_size=self.image_size, for_cnn=False)
        self.step_size = step_size
        self.num_layers = num_layers
        self.num_heads = num_heads

    # Training Setup
    def train(self, verbose=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        
        diffusion = Diffusion(timesteps=self.   timesteps)
        model = DiffusionTransformer(pixel_dim=self.image_size**2, embed_dim=self.embed_dim, num_heads=self.num_heads, num_layers=self.num_layers).to(device)

        # print the number of parameters in the mdoel
        print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")

        # wrap this in a step scheduler
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma_)

        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0
            for x_cond, x_target in dataloader:
                x_cond, x_target = x_cond.to(device), x_target.to(device)
                batch_size = x_cond.shape[0]
                
                t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).float()
                
                noise = torch.randn_like(x_target)
                x_noisy = diffusion.q_sample(x_target, t.long(), noise)
                
                noise_pred = model(x_noisy, t, x_cond)
                
                loss = nn.MSELoss()(noise_pred, noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
            
            # Print detailed info every 100 epochs
            if epoch % 100 == 0:
                avg_loss = running_loss / len(dataloader)
                if verbose:
                    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
            
        # save the model
        torch.save(model.state_dict(), f"model_{epoch}.pth")
            
        return model
    
    def plot_performance(self, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Plot predictions for all samples in dataset
        plt.figure(figsize=(12, 8))  # Adjusted height for 4 rows
        num_samples = min(5, len(self.dataset))  # Show up to 5 examples

        for i in range(num_samples):
            x_cond, x_target = self.dataset[i]
            x_cond = x_cond.unsqueeze(0).to(device)
            
            # Generate sample from model (assuming model has a sample method)
            x_sample = model.sample(x_cond, timesteps=self.timesteps)
            
            # Convert tensors to numpy arrays and ensure consistent 0-1 scale
            x_cond = x_cond.squeeze().cpu().numpy()
            x_sample = x_sample.squeeze().cpu().numpy()
            x_target = x_target.numpy()
            
            # Plot input condition (Row 1)
            plt.subplot(4, num_samples, i + 1)
            plt.imshow(x_cond.reshape(self.image_size, self.image_size), cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            if i == 0:
                plt.title('Input')
            
            # Plot model prediction (Row 2)
            plt.subplot(4, num_samples, i + 1 + num_samples)
            plt.imshow(x_sample.reshape(self.image_size, self.image_size), cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            if i == 0:
                plt.title('Prediction')
            
            # Plot ground truth (Row 3)
            plt.subplot(4, num_samples, i + 1 + 2*num_samples)
            plt.imshow(x_target.reshape(self.image_size, self.image_size), cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            if i == 0:
                plt.title('Ground Truth')

            # Plot difference between ground truth and prediction (Row 4)
            plt.subplot(4, num_samples, i + 1 + 3*num_samples)
            plt.imshow(np.abs(x_target.reshape(self.image_size, self.image_size) - x_sample.reshape(self.image_size, self.image_size)), 
                       cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            if i == 0:
                plt.title('Difference')

        plt.tight_layout()
        plt.show()

    def prediction_score(self, model):
        # evaluate the accuracy of the model on the training set, across 100 samples
        mse_list = []
        f1_scores = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # iterate through the dataset here:
        for x_cond, x_target in self.dataset:
            x_cond = x_cond.unsqueeze(0).to(device)
            x_sample = model.sample(x_cond, timesteps=self.timesteps)
            x_sample = x_sample.squeeze().cpu().numpy()
            x_target = x_target.numpy()

            # calculate the mse between the x_sample and x_target
            mse = np.mean((x_sample - x_target)**2)
            mse_list.append(mse)

            # now instead of mse calculate the F1 score, for each pixel, consider prediction of >0.5 as 1, and <0.5 as 0
            # then calculate the F1 score for each pixel, and then the average F1 score
            f1 = f1_score(np.ones_like(x_target), (np.abs(x_sample - x_target) < 0.01).astype(int))
            f1_scores.append(f1)

        return np.mean(mse_list), np.mean(f1_scores)