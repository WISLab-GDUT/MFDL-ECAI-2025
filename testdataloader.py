import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import io
import random

class ImageDataset(Dataset):
    def __init__(self, image_path, transform=None, apply_jpeg=False, jpeg_quality=50):
        self.image_path = image_path
        self.transform = transform
        self.apply_jpeg = apply_jpeg
        self.jpeg_quality = jpeg_quality
        self.image_list = []
        self.labels = []

        for subdir in os.listdir(image_path):
            subdir_path = os.path.join(image_path, subdir)
            if not os.path.isdir(subdir_path):
                continue

            real_path = os.path.join(subdir_path, '0_real')
            fake_path = os.path.join(subdir_path, '1_fake')
            if os.path.isdir(real_path) and os.path.isdir(fake_path):
                for file in os.listdir(real_path):
                    self.image_list.append(os.path.join(real_path, file))
                    self.labels.append(0)
                for file in os.listdir(fake_path):
                    self.image_list.append(os.path.join(fake_path, file))
                    self.labels.append(1)
            else:
                for category in os.listdir(subdir_path):
                    category_path = os.path.join(subdir_path, category)
                    if not os.path.isdir(category_path):
                        continue
                    real_path = os.path.join(category_path, '0_real')
                    fake_path = os.path.join(category_path, '1_fake')
                    if os.path.isdir(real_path):
                        for file in os.listdir(real_path):
                            self.image_list.append(os.path.join(real_path, file))
                            self.labels.append(0)
                    if os.path.isdir(fake_path):
                        for file in os.listdir(fake_path):
                            self.image_list.append(os.path.join(fake_path, file))
                            self.labels.append(1)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[index]

        # JPEG压缩模拟
        if self.apply_jpeg:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self.jpeg_quality)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

from torchvision.transforms import functional as F

class ConditionalResizeOrCrop:
    def __call__(self, img):
        w, h = img.size
        if w < 256 or h < 256:
            # Resize shortest side to 256, keeping aspect ratio
            img = F.resize(img, (256, 256))
        else:
            # CenterCrop to 256x256
            img = F.center_crop(img, (256, 256))
        return img



def image_dataloader(path,
                     apply_jpeg=False,
                     jpeg_quality=50,
                     flip_prob=0.5,
                     apply_blur=False,
                     blur_radius=1.0):

    transform_ops = [
        ConditionalResizeOrCrop(),  # 替代 Resize(256, 256)
        transforms.RandomHorizontalFlip(p=flip_prob),
    ]
    if apply_blur:
        transform_ops.append(transforms.GaussianBlur(kernel_size=5, sigma=blur_radius))
    transform_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform = transforms.Compose(transform_ops)

    dataset = ImageDataset(
        image_path=path,
        transform=transform,
        apply_jpeg=apply_jpeg,
        jpeg_quality=jpeg_quality
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    return dataloader

