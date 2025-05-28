import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class DatasetManager:
    def __init__(self, dataset_path, batch_size=32, transform=None):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.data_loader = None
        self.dataset = self._load_data()

    def _load_data(self):
        dataset = CustomImageDataset(self.dataset_path, transform=self.transform)
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=10
        )
        return data_loader

    def get_data_loader(self):
        return self.data_loader


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label = self._get_label_from_path(img_path)
        return image, label

    def _get_label_from_path(self, path):
        return os.path.basename(os.path.dirname(path))
