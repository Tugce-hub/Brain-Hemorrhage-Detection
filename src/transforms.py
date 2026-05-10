"""
transforms.py — Veri dönüşüm pipeline'ları.

Train: Augmentation (döndürme, flip, parlaklık) + Normalize
Val/Test: Sadece Resize + Normalize
"""

from torchvision import transforms

# ImageNet istatistikleri (transfer learning için standart)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224  # ResNet50 ve MyCNN her ikisi de 224×224 kullanır


def get_train_transform() -> transforms.Compose:
    """
    Eğitim veri seti için augmentation pipeline.

    Uygulanan teknikler:
    - RandomHorizontalFlip: BT görüntülerinde simetrik yapı
    - RandomVerticalFlip:   Farklı kesit açıları simüle eder
    - RandomRotation(±15°): Hasta konumu değişkenliği
    - ColorJitter:          Kontrast & parlaklık farklılıkları (farklı cihazlar)
    - RandomAffine:         Küçük ölçek/kaydırma varyasyonları
    - Normalize:            ImageNet ön-eğitimi için standartlaştırma
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.1,
            hue=0.05
        ),
        transforms.RandomAutocontrast(p=0.2),         # Yeni: Dinamik piksel dengesini değiştir
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2), # Yeni: Kenarları sertleştir
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.80, 2.5)  # Yeni: Kanamalari simulasyon amaciyla devasa boyutlarda goster (Zoom in)
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform() -> transforms.Compose:
    """
    Doğrulama ve test veri seti için dönüşüm pipeline (augmentation YOK).
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transform() -> transforms.Compose:
    """
    Gradio arayüzü için tek görüntü inference dönüşümü.
    """
    return get_val_transform()
