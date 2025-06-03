from datasetManager import DatasetManager
from deepBeliefNetwork import DeepBeliefNetwork
import torch
import matplotlib.pyplot as plt
import numpy as np

def show_images(imgs, title):
    # funcion pa mostrar las imagenes
    fig, axes = plt.subplots(1, len(imgs), figsize=(15, 5))
    for i, ax in enumerate(axes):
        # convertimos el tensor a numpy
        img_np = imgs[i].detach().permute(1, 2, 0).cpu().numpy()

        img_np = np.clip(img_np, 0, 1)
        # si es RGB lo mostramos normal, si no en escala de grises
        if img_np.shape[-1] == 3:
            ax.imshow(img_np)
        else:
            ax.imshow(img_np, cmap='gray')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

# vemos si tenemos GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # cargamos el dataset
    dataset_manager = DatasetManager(dataset_path="./test", batch_size=32)
    data_loader = dataset_manager.dataset

    # creamos la red
    dbn = DeepBeliefNetwork(
        in_channels=3,
        layer_sizes=[32, 64, 128],  # 3 capas con mas features cada vez
        kernel_size=5
    ).to(device)

    # entrenamos todo gracias chat gpt
    print("empezando el entrenamiento...")
    dbn.train_all_layers(
        data_loader=data_loader,
        num_epochs=50,
        lr=0.01,
        device=device
    )
    print("listo el entrenamiento!")

    # mostramos los resultados finales
    batch, _ = next(iter(data_loader))
    batch = batch.to(device)
    original_batch = batch.clone()
    batch = batch / 255.0  # normalizamos
    h_prob, h_sample = dbn.sample_h(batch)
    v_prob, v_sample = dbn.sample_v(h_sample)
    show_images(original_batch / 255.0, "Originales - Final")
    show_images(v_prob, "Reconstruidas - Final")