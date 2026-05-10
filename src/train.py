"""
train.py — Eğitim döngüsü, Early Stopping ve model kaydetme.

Özellikler:
- Early Stopping (validation loss patience ile)
- ReduceLROnPlateau learning rate scheduler
- Otomatik GPU/CPU seçimi
- Her epoch sonunda loss & accuracy loglama
- En iyi model ağırlıklarını .pth olarak kaydetme
"""

import os
import sys
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

# Proje kök dizinini Python path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import print_separator, plot_training_history, ensure_dir


# --- Early Stopping -----------------------------------------------------------
class EarlyStopping:
    """
    Validation loss iyileşmediğinde eğitimi durdurur.

    Args:
        patience (int):  Kaç epoch bekleneceği
        min_delta (float): İyileşme için minimum fark
        verbose (bool):  Durum mesajı yazdır
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, verbose: bool = True):
        self.patience   = patience
        self.min_delta  = min_delta
        self.verbose    = verbose
        self.counter    = 0
        self.best_loss  = float("inf")
        self.best_epoch = 0
        self.stop       = False

    def __call__(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"     [WAIT] Early Stopping: {self.counter}/{self.patience} "
                      f"(en iyi epoch: {self.best_epoch})")
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# --- Tek Epoch Eğitim ---------------------------------------------------------
def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  optim.Optimizer,
    device:     torch.device,
    scaler=None,
    label_smoothing: float = 0.0,
) -> tuple:
    """
    Bir epoch boyunca modeli eğitir.

    Returns:
        (avg_loss, accuracy)  accuracy 0–1 arası
    """
    model.train()
    total_loss   = 0.0
    correct      = 0
    total        = 0

    pbar = tqdm(loader, desc="  Eğitim", leave=False, dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()
        
        # Label Smoothing uygula
        smoothed_labels = labels
        if label_smoothing > 0:
             smoothed_labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss    = criterion(outputs, smoothed_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss    = criterion(outputs, smoothed_labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

        # BCEWithLogitsLoss -> sigmoid uygula, threshold 0.5
        preds    = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / total, correct / total


# --- Değerlendirme Döngüsü ----------------------------------------------------
@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple:
    """
    Val/Test seti üzerinde loss ve accuracy hesaplar.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    pbar = tqdm(loader, desc="  Doğrulama", leave=False, dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds    = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / total, correct / total


# --- Ana Eğitim Fonksiyonu ----------------------------------------------------
def train_model(
    model:           nn.Module,
    train_loader:    DataLoader,
    val_loader:      DataLoader,
    model_name:      str,
    save_dir:        str   = "outputs",
    num_epochs:      int   = 50,
    learning_rate:   float = 1e-3,
    weight_decay:    float = 1e-4,
    patience:        int   = 10,
    use_amp:         bool  = True,
    label_smoothing: float = 0.0,
    scheduler_type:  str   = "plateau",
) -> dict:
    """
    Modeli eğitir, en iyi ağırlıkları kaydeder ve geçmişi döndürür.

    Args:
        model:         PyTorch modeli
        train_loader:  Eğitim DataLoader
        val_loader:    Doğrulama DataLoader
        model_name:    Kayıt dosyası ve grafik için isim (ör. "resnet50", "mycnn")
        save_dir:      Model ve grafiklerin kaydedileceği klasör
        num_epochs:    Maksimum epoch sayısı
        learning_rate: Başlangıç öğrenme hızı
        weight_decay:  L2 regularizasyon katsayısı
        patience:      Early Stopping sabırlılığı
        use_amp:       CUDA varsa Automatic Mixed Precision kullan
        label_smoothing: BCEWithLogitsLoss icin etiket yumusatma orani
        scheduler_type: 'plateau' veya 'cosine'

    Returns:
        history = {
            'train_loss': [...], 'val_loss': [...],
            'train_acc':  [...], 'val_acc':  [...],
            'best_epoch': int,  'best_val_loss': float,
            'save_path':  str
        }
    """
    ensure_dir(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    print_separator(f"  {model_name.upper()} EĞİTİMİ BAŞLIYOR  ")
    print(f"  Cihaz:  {device}")
    print(f"  Epoch:  {num_epochs}  |  LR: {learning_rate}  |  Patience: {patience}")
    print_separator()

    # PyTorch'un BCEWithLogitsLoss fonksiyonu dogrudan label_smoothing desteklemez (sadece CrossEntropyLoss destekler).
    # O yuzden ozel bir fonksiyon yazmiyoruz, manuel sekilde asagida uygulayacagiz.
    criterion = nn.BCEWithLogitsLoss()
    optimizer     = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7, verbose=True
        )
        
    early_stopper = EarlyStopping(patience=patience, verbose=True)

    # AMP scaler (sadece CUDA'da çalışır)
    scaler = None
    if use_amp and torch.cuda.is_available():
        scaler = torch.amp.GradScaler()

    # Geçmiş
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_loss  = float("inf")
    best_weights   = None
    best_epoch     = 0

    total_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # -- Eğitim ----------------------------------------------------------
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, label_smoothing
        )

        # -- Doğrulama -------------------------------------------------------
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # -- Scheduler -------------------------------------------------------
        if scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        # -- Geçmişe kaydet --------------------------------------------------
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        print(
            f"  Epoch [{epoch:3d}/{num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.1f}%  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.1f}%  "
            f"({epoch_time:.1f}s)"
        )

        # -- En iyi model kaydet ---------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = copy.deepcopy(model.state_dict())
            best_epoch    = epoch
            print(f"     [SAVE] Yeni en iyi model kaydedildi (Val Loss: {best_val_loss:.4f})")

        # -- Early Stopping kontrol ------------------------------------------
        if early_stopper(val_loss, epoch):
            print(f"     [STOP] Early Stopping tetiklendi! Epoch {epoch}")
            print(f"     En iyi epoch: {best_epoch}, Val Loss: {best_val_loss:.4f}")
            break

    # -- En iyi ağırlıkları yükle ve kaydet -----------------------------------
    model.load_state_dict(best_weights)
    save_path = os.path.join(save_dir, f"{model_name}_best.pth")
    torch.save({
        "model_state_dict": best_weights,
        "model_name":       model_name,
        "best_epoch":       best_epoch,
        "best_val_loss":    best_val_loss,
        "history":          history,
    }, save_path)

    total_time = time.time() - total_start
    print_separator()
    print(f"  [OK] Eğitim tamamlandı! Toplam süre: {total_time/60:.1f} dakika")
    print(f"  [SAVE] Model kaydedildi: {save_path}")
    print_separator()

    # -- Grafik çiz -----------------------------------------------------------
    plot_training_history(
        history["train_loss"], history["val_loss"],
        history["train_acc"],  history["val_acc"],
        model_name=model_name,
        save_dir=save_dir
    )

    history["best_epoch"]    = best_epoch
    history["best_val_loss"] = best_val_loss
    history["save_path"]     = save_path

    return history


# --- Test --------------------------------------------------------------------
if __name__ == "__main__":
    print("train.py modülü hazır. train_all.py üzerinden çalıştırın.")
