"""
dataset.py — Head CT Veri Seti tanımı ve DataLoader üretimi.

Yapı:
- labels.csv okur (id, hemorrhage sütunları)
- id'den dosya adı üretir: 000.png, 001.png, ...
- Stratified Train/Val/Test split (70/15/15)
- PyTorch Dataset ve DataLoader döndürür
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from src.transforms import get_train_transform, get_val_transform


# --- Sabitler ----------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "head_ct", "head_ct")
LABELS_CSV = os.path.join(BASE_DIR, "labels.csv")


# --- Dataset Sınıfı ----------------------------------------------------------
class HeadCTDataset(Dataset):
    """
    Head CT Kanama Tespiti Veri Seti.

    Args:
        df (pd.DataFrame): 'filename' ve 'label' sütunlarını içeren DataFrame
        images_dir (str):  Görüntülerin bulunduğu klasör yolu
        transform:         torchvision transform pipeline
    """

    def __init__(self, df: pd.DataFrame, images_dir: str, transform=None):
        self.df         = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform  = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["filename"])

        # Görüntüyü aç ve RGB'ye çevir (CT grayscale olabilir)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["label"], dtype=torch.float32)
        return image, label


# --- Yardımcı: CSV -> DataFrame -----------------------------------------------
def load_dataframe(labels_csv: str = LABELS_CSV) -> pd.DataFrame:
    """
    labels.csv -> temizlenmiş DataFrame.

    Sütun adlarındaki boşlukları temizler, dosya adını üretir.
    id=0 -> '000.png', id=1 -> '001.png', ...
    """
    df = pd.read_csv(labels_csv)

    # Sütun adlarındaki boşlukları temizle (CSV'de ' hemorrhage' var)
    df.columns = df.columns.str.strip()

    # Dosya adını üret: id'yi 3 haneli string'e çevir
    df["filename"] = df["id"].apply(lambda x: f"{int(x):03d}.png")
    df["label"]    = df["hemorrhage"].astype(int)

    return df[["filename", "label"]]


# --- Ana Fonksiyon: DataLoader'ları Döndür -----------------------------------
def get_dataloaders(
    labels_csv:  str   = LABELS_CSV,
    images_dir:  str   = IMAGES_DIR,
    batch_size:  int   = 16,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    random_seed: int   = 42,
    num_workers: int   = 0,
) -> dict:
    """
    Train / Val / Test DataLoader'larını oluşturur ve döndürür.

    Args:
        labels_csv:  labels.csv dosyasının yolu
        images_dir:  Görüntü klasörü yolu
        batch_size:  Mini-batch büyüklüğü
        val_ratio:   Doğrulama oranı (0–1)
        test_ratio:  Test oranı (0–1)
        random_seed: Tekrar üretilebilirlik için seed
        num_workers: DataLoader worker sayısı (Windows'ta 0 önerilir)

    Returns:
        {
          'train': DataLoader,
          'val':   DataLoader,
          'test':  DataLoader,
          'class_counts': {'train': {...}, 'val': {...}, 'test': {...}}
        }
    """
    df = load_dataframe(labels_csv)

    # -- Adım 1: Test setini ayır ---------------------------------------------
    df_trainval, df_test = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["label"],
        random_state=random_seed
    )

    # -- Adım 2: Train / Val ayır ---------------------------------------------
    # val_ratio, orijinal veri setinin oranı olmalı -> trainval içindeki oran hesaplanır
    relative_val = val_ratio / (1.0 - test_ratio)

    df_train, df_val = train_test_split(
        df_trainval,
        test_size=relative_val,
        stratify=df_trainval["label"],
        random_state=random_seed
    )

    # -- Adım 3: Dataset & DataLoader -----------------------------------------
    train_transform = get_train_transform()
    val_transform   = get_val_transform()

    train_ds = HeadCTDataset(df_train, images_dir, transform=train_transform)
    val_ds   = HeadCTDataset(df_val,   images_dir, transform=val_transform)
    test_ds  = HeadCTDataset(df_test,  images_dir, transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    # -- Adım 4: Bilgi yazdır -------------------------------------------------
    def class_info(split_df: pd.DataFrame, name: str) -> dict:
        counts = split_df["label"].value_counts().to_dict()
        normal     = counts.get(0, 0)
        hemorrhage = counts.get(1, 0)
        print(
            f"  {name:6s} -> {len(split_df):3d} görüntü  |  "
            f"Normal: {normal}  |  Kanamalı: {hemorrhage}"
        )
        return {"normal": normal, "hemorrhage": hemorrhage}

    print("\n[INFO] Veri Seti Bölünmesi:")
    print("-" * 55)
    info = {
        "train": class_info(df_train, "Train"),
        "val":   class_info(df_val,   "Val"),
        "test":  class_info(df_test,  "Test"),
    }
    print("-" * 55)

    return {
        "train":        train_loader,
        "val":          val_loader,
        "test":         test_loader,
        "class_counts": info,
        "df_train":     df_train,
        "df_val":       df_val,
        "df_test":      df_test,
    }


# --- Test --------------------------------------------------------------------
if __name__ == "__main__":
    loaders = get_dataloaders(batch_size=8)
    train_loader = loaders["train"]
    images, labels = next(iter(train_loader))
    print(f"\nBir batch: images={images.shape}, labels={labels.shape}")
    print(f"Label dağılımı (bu batch): {labels.tolist()}")
