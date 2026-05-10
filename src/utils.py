"""
utils.py — Yardımcı fonksiyonlar: grafik çizimi, klasör oluşturma vb.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Headless ortamlarda hata vermemesi için
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def ensure_dir(path: str) -> str:
    """Klasör yoksa oluştur, yolunu döndür."""
    os.makedirs(path, exist_ok=True)
    return path


def plot_training_history(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    model_name: str,
    save_dir: str = "outputs"
) -> str:
    """
    Eğitim ve doğrulama loss/accuracy grafiklerini çizer ve PNG olarak kaydeder.

    Args:
        train_losses: Her epoch için eğitim loss listesi
        val_losses:   Her epoch için doğrulama loss listesi
        train_accs:   Her epoch için eğitim accuracy listesi (0–1 arası)
        val_accs:     Her epoch için doğrulama accuracy listesi (0–1 arası)
        model_name:   Dosya adında kullanılacak model ismi
        save_dir:     PNG'lerin kaydedileceği klasör

    Returns:
        Kaydedilen dosyanın tam yolu
    """
    ensure_dir(save_dir)
    epochs = range(1, len(train_losses) + 1)

    # --- Stil ----------------------------------------------------------------
    plt.style.use("dark_background")
    ACCENT_TRAIN = "#4FC3F7"   # açık mavi
    ACCENT_VAL   = "#EF9A9A"   # açık kırmızı
    BG_COLOR     = "#1A1A2E"
    GRID_COLOR   = "#2D2D44"
    TEXT_COLOR   = "#E0E0E0"

    fig = plt.figure(figsize=(14, 6), facecolor=BG_COLOR)
    fig.suptitle(
        f"{model_name} — Eğitim Geçmişi",
        fontsize=16, fontweight="bold", color=TEXT_COLOR, y=1.02
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # --- Loss Grafiği --------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG_COLOR)
    ax1.plot(epochs, train_losses, color=ACCENT_TRAIN, linewidth=2.2,
             label="Train Loss", marker="o", markersize=4)
    ax1.plot(epochs, val_losses, color=ACCENT_VAL, linewidth=2.2,
             label="Val Loss", marker="s", markersize=4, linestyle="--")
    # En iyi val loss noktasını işaretle
    best_epoch = int(np.argmin(val_losses)) + 1
    ax1.axvline(x=best_epoch, color="#FFF176", linestyle=":", linewidth=1.5,
                label=f"Best Epoch ({best_epoch})")
    ax1.set_title("Loss", color=TEXT_COLOR, fontsize=13)
    ax1.set_xlabel("Epoch", color=TEXT_COLOR)
    ax1.set_ylabel("Loss Değeri", color=TEXT_COLOR)
    ax1.tick_params(colors=TEXT_COLOR)
    ax1.grid(color=GRID_COLOR, linestyle="--", linewidth=0.7)
    ax1.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax1.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    # --- Accuracy Grafiği ----------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG_COLOR)
    train_accs_pct = [a * 100 for a in train_accs]
    val_accs_pct   = [a * 100 for a in val_accs]
    ax2.plot(epochs, train_accs_pct, color=ACCENT_TRAIN, linewidth=2.2,
             label="Train Acc", marker="o", markersize=4)
    ax2.plot(epochs, val_accs_pct, color=ACCENT_VAL, linewidth=2.2,
             label="Val Acc", marker="s", markersize=4, linestyle="--")
    best_acc_epoch = int(np.argmax(val_accs_pct)) + 1
    ax2.axvline(x=best_acc_epoch, color="#FFF176", linestyle=":", linewidth=1.5,
                label=f"Best Epoch ({best_acc_epoch})")
    ax2.set_title("Accuracy", color=TEXT_COLOR, fontsize=13)
    ax2.set_xlabel("Epoch", color=TEXT_COLOR)
    ax2.set_ylabel("Doğruluk (%)", color=TEXT_COLOR)
    ax2.set_ylim(0, 105)
    ax2.tick_params(colors=TEXT_COLOR)
    ax2.grid(color=GRID_COLOR, linestyle="--", linewidth=0.7)
    ax2.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_history.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [OK] Grafik kaydedildi: {save_path}")
    return save_path


def print_separator(title: str = "", width: int = 60) -> None:
    """Konsola görsel bölücü yazdır."""
    if title:
        pad = (width - len(title) - 2) // 2
        print("-" * pad + f" {title} " + "-" * pad)
    else:
        print("-" * width)
