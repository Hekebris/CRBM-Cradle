from datasetManager import DatasetManager
from crbmConvolutionalLayer import CRBMConvolutionalLayer
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset_manager = DatasetManager(dataset_path="./test", batch_size=64)
    data_loader = dataset_manager.dataset

    crbm_layer = CRBMConvolutionalLayer(
        in_channels=3,
        out_channels=5,
        kernel_size=2,
        pool_size=2,
    )

    crbm_layer2 = CRBMConvolutionalLayer(
        in_channels=5,
        out_channels=10,
        kernel_size=2,
        pool_size=2,
    )

    crbm_layer3 = CRBMConvolutionalLayer(
        in_channels=10,
        out_channels=20,
        kernel_size=2,
        pool_size=2,
    )

    crbm_layer4 = CRBMConvolutionalLayer(
        in_channels=20,
        out_channels=40,
        kernel_size=2,
        pool_size=2,
    )

    for images, labels in data_loader:
        ####CONV LAYER 1####
        output = crbm_layer(images)

        print("Shape del output:", output.shape)

        output = torch.clamp(output, 0, 1)

        filters = output[0].detach().cpu().numpy()

        fig, axes = plt.subplots(1, 5, figsize=(15, 5))
        for i in range(5):
            axes[i].imshow(filters[i])
            axes[i].axis("off")
        plt.show()

        ####CONV LAYER 2####
        output = crbm_layer2(output)
        print("Shape del output:", output.shape)

        output = torch.clamp(output, 0, 1)

        filters = output[0].detach().cpu().numpy()

        fig, axes = plt.subplots(1, 10, figsize=(15, 5))
        for i in range(10):
            axes[i].imshow(filters[i])
            axes[i].axis("off")
        plt.show()

        ####CONV LAYER 3####
        output = crbm_layer3(output)
        print("Shape del output:", output.shape)

        output = torch.clamp(output, 0, 1)

        filters = output[0].detach().cpu().numpy()

        fig, axes = plt.subplots(1, 20, figsize=(15, 5))
        for i in range(20):
            axes[i].imshow(filters[i])
            axes[i].axis("off")
        plt.show()

        ####CONV LAYER 4####
        output = crbm_layer4(output)
        print("Shape del output:", output.shape)

        output = torch.clamp(output, 0, 1)

        filters = output[0].detach().cpu().numpy()

        fig, axes = plt.subplots(1, 40, figsize=(15, 5))
        for i in range(40):
            axes[i].imshow(filters[i])
            axes[i].axis("off")
        plt.show()

        break
