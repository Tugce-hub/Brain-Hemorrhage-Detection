"""
evaluate.py -- Test seti degerlendirmesi.

Hesaplanan metrikler:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (seaborn heatmap -> PNG)
- Her iki modeli karsilastiran ozet tablo

Kullanim:
    python src/evaluate.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset  import get_dataloaders
from src.model_pretrained import build_resnet50
from src.model_custom     import build_mycnn
from src.utils    import ensure_dir, print_separator


# ─── Sabitler ────────────────────────────────────────────────────────────────
OUTPUTS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
CLASS_NAMES  = ["Normal", "Kanamalı"]


# ─── Tahmin Üret ─────────────────────────────────────────────────────────────
@torch.no_grad()
def get_predictions(model: nn.Module, loader, device: torch.device) -> tuple:
    """
    Test loader üzerinde tahmin ve etiket listesi üret.

    Returns:
        (all_preds, all_labels, all_probs) : numpy array'ler
    """
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)                              # (B, 1) logit
        probs   = torch.sigmoid(outputs).squeeze(1)         # (B,) 0–1
        preds   = (probs >= 0.5).long()

        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return (
        np.array(all_preds),
        np.array(all_labels, dtype=int),
        np.array(all_probs)
    )


# ─── Confusion Matrix Çiz ────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: str = OUTPUTS_DIR
) -> str:
    """
    Confusion Matrix'i seaborn heatmap ile görselleştir ve PNG kaydet.
    """
    ensure_dir(save_dir)
    cm = confusion_matrix(y_true, y_pred)

    BG_COLOR   = "#1A1A2E"
    CMAP       = sns.color_palette("Blues", as_cmap=True)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=CMAP,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=1,
        linecolor="#2D2D44",
        ax=ax,
        annot_kws={"size": 20, "weight": "bold", "color": "white"},
    )

    ax.set_title(f"{model_name} — Confusion Matrix", color="white",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Tahmin Edilen", color="white", fontsize=12)
    ax.set_ylabel("Gerçek", color="white", fontsize=12)
    ax.tick_params(colors="white", labelsize=11)

    # Colorbar rengini de ayarla
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="white")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  ✅ Confusion Matrix kaydedildi: {save_path}")
    return save_path


# ─── Model Yükle ─────────────────────────────────────────────────────────────
def load_model(model_name: str, device: torch.device) -> nn.Module:
    """Kaydedilmiş checkpoint'ten model yükle."""
    ckpt_path = os.path.join(OUTPUTS_DIR, f"{model_name}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Model bulunamadı: {ckpt_path}\n"
            f"Önce 'python train_all.py' komutunu çalıştırın."
        )

    if model_name == "resnet50":
        model = build_resnet50(freeze_backbone=False)
    elif model_name == "mycnn":
        model = build_mycnn()
    else:
        raise ValueError(f"Bilinmeyen model adı: {model_name}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    best_epoch    = checkpoint.get("best_epoch", "?")
    best_val_loss = checkpoint.get("best_val_loss", float("nan"))
    print(f"  [OK] '{model_name}' yuklendi -- Best Epoch: {best_epoch}, Val Loss: {best_val_loss:.4f}")
    return model


# ─── Tek Model Değerlendirme ──────────────────────────────────────────────────
def evaluate_model(
    model_name: str,
    test_loader,
    device: torch.device,
    save_dir: str = OUTPUTS_DIR,
) -> dict:
    """
    Model yükle → test seti tahmin → metrikleri hesapla → ekrana yazdır.

    Returns:
        metrics dict
    """
    print_separator(f" {model_name.upper()} TEST DEĞERLENDİRMESİ ")

    model               = load_model(model_name, device)
    y_pred, y_true, _   = get_predictions(model, test_loader, device)

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    # ── Konsol Çıktısı ───────────────────────────────────────────────────────
    print(f"\n  [SONUC] Test Seti Metrikleri ({model_name}):")
    print(f"  {'─'*40}")
    print(f"  {'Accuracy ':.<30} {accuracy*100:>6.2f}%")
    print(f"  {'Precision':.<30} {precision*100:>6.2f}%")
    print(f"  {'Recall':.<30} {recall*100:>6.2f}%")
    print(f"  {'F1-Score':.<30} {f1*100:>6.2f}%")
    print(f"  {'─'*40}")
    print(f"\n  [RAPOR] Siniflandirma Raporu:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # -- Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, model_name=model_name, save_dir=save_dir)
    print_separator()

    return {
        "model":     model_name,
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


# ─── Karşılaştırma Tablosu ───────────────────────────────────────────────────
def print_comparison_table(results: list) -> None:
    """İki modelin metriklerini yan yana göster."""
    print_separator(" MODEL KARŞILAŞTIRMA TABLOSU ")
    header = f"  {'Model':15s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}"
    print(header)
    print("  " + "─" * 50)
    for r in results:
        print(
            f"  {r['model']:15s} "
            f"{r['accuracy']*100:>9.2f}% "
            f"{r['precision']*100:>9.2f}% "
            f"{r['recall']*100:>9.2f}% "
            f"{r['f1']*100:>9.2f}%"
        )
    print_separator()


# ─── Ana Çalışma Bloğu ───────────────────────────────────────────────────────
if __name__ == "__main__":
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[CIHAZ] Degerlendirme: {device}")

    # Veri yukle (sadece test seti gerekli)
    loaders     = get_dataloaders(batch_size=16)
    test_loader = loaders["test"]

    results = []
    for name in ["resnet50", "mycnn"]:
        try:
            r = evaluate_model(name, test_loader, device, save_dir=OUTPUTS_DIR)
            results.append(r)
        except FileNotFoundError as e:
            print(f"  [UYARI] {e}")

    if len(results) > 1:
        print_comparison_table(results)

    print("\n[TAMAM] Degerlendirme tamamlandi!")
    print(f"   Grafik ve matrisler: {OUTPUTS_DIR}/")
