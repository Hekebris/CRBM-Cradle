import torch
import torch.nn as nn
from crbmConvolutionalLayer import CRBM

class DeepBeliefNetwork(nn.Module):
    def __init__(self, in_channels=3, layer_sizes=[32, 64, 128], kernel_size=5):
        super(DeepBeliefNetwork, self).__init__()
        
        # aca guardamos las capas del CRBM
        self.crbms = nn.ModuleList()
        prev_channels = in_channels
        
        for out_channels in layer_sizes:
            self.crbms.append(CRBM(
                in_channels=prev_channels,
                out_channels=out_channels,
                kernel_size=kernel_size
            ))
            prev_channels = out_channels

    def forward(self, x):
        h = x
        for crbm in self.crbms:
            h_prob, h_sample = crbm.sample_h(h)
            h = h_sample
        return h

    def train_layer(self, layer_idx, data_loader, num_epochs=50, lr=0.01, device='cuda'):
        crbm = self.crbms[layer_idx]
        crbm = crbm.to(device)
        
        initial_lr = lr
        min_lr = lr * 0.1
        lr_decay = 0.95
        current_lr = initial_lr
        
        print(f"\nTraining CRBM layer {layer_idx + 1}/{len(self.crbms)}")
        
        # si no es la primera capa, precalculamos todo pa no hacerlo mil veces
        if layer_idx > 0:
            print("calculando features de las capas anteriores...")
            processed_data = []
            with torch.no_grad():
                for batch, _ in data_loader:
                    batch = batch.to(device) / 255.0
                    h = batch
                    for i in range(layer_idx):
                        h_prob, h_sample = self.crbms[i].sample_h(h)
                        h = h_sample
                    processed_data.append(h)
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            if layer_idx > 0:
                for batch in processed_data:
                    loss = crbm.contrastive_divergence(batch, lr=current_lr)
                    total_loss += loss.item()
                    num_batches += 1
            else:
                for batch, _ in data_loader:
                    batch = batch.to(device) / 255.0
                    loss = crbm.contrastive_divergence(batch, lr=current_lr)
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"[CRBM{layer_idx+1}] Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")
            
            current_lr = max(min_lr, current_lr * lr_decay)
            
            if (epoch + 1) % 50 == 0:
                with torch.no_grad():
                    if layer_idx > 0:
                        batch = processed_data[0]
                    else:
                        batch, _ = next(iter(data_loader))
                        batch = batch.to(device) / 255.0
                    
                    h_prob, h_sample = crbm.sample_h(batch)
                    v_prob, v_sample = crbm.sample_v(h_sample)
                    
                    if batch.shape[1] > 3:
                        vis_batch = batch[:, :3]
                        vis_v_prob = v_prob[:, :3]
                    else:
                        vis_batch = batch
                        vis_v_prob = v_prob
                    
                    self.show_images(vis_batch, f"Originales - Layer {layer_idx+1} - Epoch {epoch+1}")
                    self.show_images(vis_v_prob, f"Reconstruidas - Layer {layer_idx+1} - Epoch {epoch+1}")

    def show_images(self, imgs, title):
        import matplotlib.pyplot as plt
        import numpy as np
        
        n_images = min(5, len(imgs))
        imgs = imgs[:n_images]
        
        fig, axes = plt.subplots(1, n_images, figsize=(15, 5))
        for i, ax in enumerate(axes):
            img_np = imgs[i].detach().permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            ax.imshow(img_np)
            ax.axis('off')
        plt.suptitle(title)
        plt.show()

    def train_all_layers(self, data_loader, num_epochs=50, lr=0.01, device='cuda'):
        for i in range(len(self.crbms)):
            self.train_layer(i, data_loader, num_epochs, lr, device) 