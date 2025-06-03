import torch
import torch.nn as nn
import torch.nn.functional as F

class CRBM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CRBM, self).__init__()
        # Initialize weights with smaller values for better stability
        self.W = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001)
        self.v_bias = nn.Parameter(torch.zeros(in_channels))
        self.h_bias = nn.Parameter(torch.zeros(out_channels))

    def sample_h(self, v):
        h_act = F.conv2d(v, self.W, bias=self.h_bias)
        h_prob = torch.sigmoid(h_act)
        h_sample = torch.bernoulli(h_prob)
        return h_prob, h_sample

    def sample_v(self, h):
        v_act = F.conv_transpose2d(h, self.W, bias=self.v_bias)
        v_prob = torch.sigmoid(v_act)
        v_sample = torch.bernoulli(v_prob)
        return v_prob, v_sample

    def contrastive_divergence(self, v, lr=0.01):
        # Single step CD
        h_prob0, h_sample0 = self.sample_h(v)
        v_prob1, v_sample1 = self.sample_v(h_sample0)
        h_prob1, _ = self.sample_h(v_sample1)

        # Update weights
        for j in range(self.W.size(0)):
            # Get the j-th filter's hidden probabilities
            h_prob0_j = h_prob0[:, j:j+1]  # [batch, 1, H, W]
            h_prob1_j = h_prob1[:, j:j+1]  # [batch, 1, H, W]
            
            # Compute positive phase
            positive = torch.zeros_like(self.W[j])
            for i in range(v.size(0)):
                positive += F.conv2d(
                    v[i:i+1],  # [1, in_channels, H, W]
                    h_prob0_j[i:i+1].repeat(1, v.size(1), 1, 1)  # [1, in_channels, H, W]
                )[0]  # [in_channels, kernel_size, kernel_size]
            
            # Compute negative phase
            negative = torch.zeros_like(self.W[j])
            for i in range(v_prob1.size(0)):
                negative += F.conv2d(
                    v_prob1[i:i+1],  # [1, in_channels, H, W]
                    h_prob1_j[i:i+1].repeat(1, v.size(1), 1, 1)  # [1, in_channels, H, W]
                )[0]  # [in_channels, kernel_size, kernel_size]
            
            # Average over batch and update
            self.W.data[j] += lr * (positive - negative) / v.size(0)

        # Update biases
        self.v_bias.data += lr * torch.mean(v - v_sample1, dim=(0, 2, 3))
        self.h_bias.data += lr * torch.mean(h_prob0 - h_prob1, dim=(0, 2, 3))

        return F.mse_loss(v_sample1, v)

