"""
train_kfold.py — K-Fold Cross Validation ile Her Iki Modeli Egit.

Strateji:
  1. Tum veri seti (200 gorsel) K parcaya bolunur (K=5)
  2. Her fold'da:
      - K-1 parca Train, 1 parca Validation olarak kullanilir
      - Model sifirdan baslatilir, eğitilir
      - En iyi val_loss'taki agirliklari kaydedilir
  3. K fold sonunda:
      - Her fold'un test accuracy'si ortalamayla raporlanir
      - En iyi fold'un modeli 'best overall' olarak kaydedilir

Neden K-Fold?
  - 200 gorsellik kucuk veri setinde tek bir split'e gore
    model degerlendirmesi guvenilmez olabilir.
  - K-Fold, veri setinin TAMAMINI hem egitim hem val icin kullanir.
  - Sonuclar daha kararli ve gercekci.

Kullanim:
    python train_kfold.py
"""

import os
import sys
import copy
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.dataset          import load_dataframe, HeadCTDataset
from src.transforms       import get_train_transform, get_val_transform
from src.model_pretrained import build_resnet50
from src.model_custom     import build_mycnn
from src.train            import train_one_epoch, evaluate, EarlyStopping
from src.evaluate         import plot_confusion_matrix, OUTPUTS_DIR
from src.utils            import ensure_dir, print_separator, plot_training_history

import warnings
warnings.filterwarnings("ignore")


# ─── Sabitler ────────────────────────────────────────────────────────────────
IMAGES_DIR  = os.path.join(ROOT, "head_ct", "head_ct")
LABELS_CSV  = os.path.join(ROOT, "labels.csv")
KFOLD_DIR   = os.path.join(ROOT, "outputs", "kfold")

# Grid search'ten bulunan BEST parametreler
BEST_CONFIG = {
    # Gorseldeki grid search: ResNet50 LR=1e-4, BS=8 -> %96.67
    "resnet50": {
        "learning_rate": 1e-3,
        "batch_size":    16,
        "weight_decay":  5e-4,
        "dropout_rate":  0.3,
        "num_epochs":    50,
        "patience":      10,
    },
    # Grid search CSV'den: MyCNN LR=1e-3, Dropout=0.3, WD=2e-4 -> %90.32 val acc
    # Gorseldeki: MyCNN LR=1e-3, BS=16 -> %93.33 test acc
    "mycnn": {
        "learning_rate": 1e-3,
        "batch_size":    16,
        "weight_decay":  2e-4,
        "dropout_rate":  0.3,
        "num_epochs":    60,
        "patience":      15,
    },
}

K_FOLDS = 5


# ─── Model Fabrikasi ─────────────────────────────────────────────────────────
def build_model(model_name: str, cfg: dict) -> nn.Module:
    if model_name == "resnet50":
        return build_resnet50(freeze_backbone=True, dropout_rate=cfg["dropout_rate"])
    else:
        return build_mycnn(dropout_rate=cfg["dropout_rate"])


# ─── Tek Model K-Fold Egitimi ─────────────────────────────────────────────────
def kfold_train(model_name: str) -> dict:
    cfg    = BEST_CONFIG[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(KFOLD_DIR)

    print_separator(f"  {model_name.upper()} — {K_FOLDS}-FOLD EGITIMI  ")
    print(f"  LR={cfg['learning_rate']}  |  BS={cfg['batch_size']}  |"
          f"  Dropout={cfg['dropout_rate']}  |  WD={cfg['weight_decay']}")
    print(f"  Cihaz: {device}\n")

    # ── Tam veri setini yukle ─────────────────────────────────────────────────
    df_all = load_dataframe(LABELS_CSV)
    X      = df_all["filename"].values
    y      = df_all["label"].values

    skf    = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    best_fold_score   = -1
    best_fold_weights = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print_separator(f" FOLD {fold_idx}/{K_FOLDS} ")

        # ── Dataset & DataLoader ──────────────────────────────────────────────
        df_train = df_all.iloc[train_idx].reset_index(drop=True)
        df_val   = df_all.iloc[val_idx].reset_index(drop=True)

        train_ds = HeadCTDataset(df_train, IMAGES_DIR, transform=get_train_transform())
        val_ds   = HeadCTDataset(df_val,   IMAGES_DIR, transform=get_val_transform())

        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg["batch_size"], shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )

        # ── Model, Optimizer, Scheduler, EarlyStopping ───────────────────────
        model     = build_model(model_name, cfg).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"], eta_min=1e-7)
        stopper   = EarlyStopping(patience=cfg["patience"], verbose=True)

        scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None

        best_val_loss  = float("inf")
        best_weights   = None
        best_epoch     = 0
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(1, cfg["num_epochs"] + 1):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )
            vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(vl_acc)

            print(f"  Fold{fold_idx} Epoch[{epoch:3d}/{cfg['num_epochs']}]  "
                  f"TrLoss:{tr_loss:.4f} TrAcc:{tr_acc*100:.1f}%  |  "
                  f"ValLoss:{vl_loss:.4f} ValAcc:{vl_acc*100:.1f}%")

            if vl_loss < best_val_loss:
                best_val_loss  = vl_loss
                best_weights   = copy.deepcopy(model.state_dict())
                best_epoch     = epoch

            if stopper(vl_loss, epoch):
                print(f"  [EarlyStop] Fold {fold_idx} durdu. En iyi epoch: {best_epoch}")
                break

        # ── En iyi agirliklari yukle, val seti uzerinde degerlendir ──────────
        model.load_state_dict(best_weights)
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs  = imgs.to(device)
                logit = model(imgs)
                preds = (torch.sigmoid(logit).squeeze(1) >= 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labs.numpy().astype(int))

        acc  = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec  = recall_score(all_labels, all_preds, zero_division=0)
        f1   = f1_score(all_labels, all_preds, zero_division=0)

        print(f"\n  [Fold {fold_idx} Sonuc]  "
              f"Acc={acc*100:.2f}%  Prec={prec*100:.2f}%  "
              f"Rec={rec*100:.2f}%  F1={f1*100:.2f}%")

        fold_results.append({
            "fold": fold_idx,
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        })

        # Egitim grafigini kaydet
        plot_training_history(
            history["train_loss"], history["val_loss"],
            history["train_acc"],  history["val_acc"],
            model_name=f"{model_name}_fold{fold_idx}",
            save_dir=KFOLD_DIR
        )

        # En iyi fold'u takip et
        if acc > best_fold_score:
            best_fold_score   = acc
            best_fold_weights = copy.deepcopy(best_weights)

    # ── Tum fold sonuclarini ozetle ─────────────────────────────────────────
    print_separator(f"  {model_name.upper()} K-FOLD OZETI  ")
    df = pd.DataFrame(fold_results)

    mean_acc  = df["accuracy"].mean()  * 100
    std_acc   = df["accuracy"].std()   * 100
    mean_f1   = df["f1"].mean()        * 100
    std_f1    = df["f1"].std()         * 100
    mean_prec = df["precision"].mean() * 100
    mean_rec  = df["recall"].mean()    * 100

    print(f"\n  {'Fold':<8} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("  " + "-" * 50)
    for _, row in df.iterrows():
        print(f"  {'Fold '+str(int(row['fold'])):<8} "
              f"{row['accuracy']*100:>9.2f}% "
              f"{row['f1']*100:>9.2f}% "
              f"{row['precision']*100:>9.2f}% "
              f"{row['recall']*100:>9.2f}%")
    print("  " + "-" * 50)
    print(f"  {'ORTALAMA':<8} {mean_acc:>9.2f}% {mean_f1:>9.2f}% "
          f"{mean_prec:>9.2f}% {mean_rec:>9.2f}%")
    print(f"  {'STD':<8} {std_acc:>9.2f}% {std_f1:>9.2f}%")
    print_separator()

    # En iyi fold agirliklarini kaydet
    save_path = os.path.join(OUTPUTS_DIR, f"{model_name}_best.pth")
    torch.save({
        "model_state_dict": best_fold_weights,
        "model_name":       model_name,
        "best_fold_score":  best_fold_score,
        "kfold_mean_acc":   mean_acc,
        "kfold_std_acc":    std_acc,
        "kfold_mean_f1":    mean_f1,
    }, save_path)
    print(f"  [KAYDEDILDI] En iyi fold agirliklari: {save_path}")

    # CSV kaydet
    csv_path = os.path.join(KFOLD_DIR, f"{model_name}_kfold_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [KAYDEDILDI] K-Fold CSV: {csv_path}")

    return {
        "model":      model_name,
        "mean_acc":   mean_acc,
        "std_acc":    std_acc,
        "mean_f1":    mean_f1,
        "std_f1":     std_f1,
        "mean_prec":  mean_prec,
        "mean_rec":   mean_rec,
        "fold_df":    df,
    }


# ─── Ana ─────────────────────────────────────────────────────────────────────
def main():
    ensure_dir(KFOLD_DIR)
    print_separator("  BEYIN KANAMASI — K-FOLD EGITIMI BASLIYOR  ")
    print(f"  K = {K_FOLDS}  |  Toplam fold: {K_FOLDS}")
    print(f"  ResNet50 best params: LR={BEST_CONFIG['resnet50']['learning_rate']}"
          f", BS={BEST_CONFIG['resnet50']['batch_size']}")
    print(f"  MyCNN best params:    LR={BEST_CONFIG['mycnn']['learning_rate']}"
          f", BS={BEST_CONFIG['mycnn']['batch_size']}"
          f", Dropout={BEST_CONFIG['mycnn']['dropout_rate']}")
    print()

    all_results = []

    # ── ResNet50 K-Fold ───────────────────────────────────────────────────────
    r1 = kfold_train("resnet50")
    all_results.append(r1)

    # ── MyCNN K-Fold ─────────────────────────────────────────────────────────
    # r2 = kfold_train("mycnn")
    # all_results.append(r2)

    # ── Final Karsilastirma ───────────────────────────────────────────────────
    print_separator("  FINAL MODEL KARSILASTIRMASI  ")
    print(f"\n  {'Model':<12} {'Ortalama Acc':>14} {'Std':>8} {'Ortalama F1':>14} {'Std':>8}")
    print("  " + "-" * 60)
    for r in all_results:
        print(f"  {r['model']:<12} {r['mean_acc']:>13.2f}% "
              f"{r['std_acc']:>7.2f}% {r['mean_f1']:>13.2f}% "
              f"{r['std_f1']:>7.2f}%")
    print("  " + "-" * 60)

    best = max(all_results, key=lambda x: x["mean_f1"])
    print(f"\n  [KAZANAN] {best['model'].upper()}")
    print(f"  Ort. F1 : {best['mean_f1']:.2f}% +/- {best['std_f1']:.2f}%")
    print(f"  Ort. Acc: {best['mean_acc']:.2f}% +/- {best['std_acc']:.2f}%")
    print(f"\n  [TAMAM] Egitim tamamlandi!")
    print(f"  - Grafik dosyalari : outputs/kfold/*.png")
    print(f"  - CSV sonuclari    : outputs/kfold/*_kfold_results.csv")
    print(f"  - Model dosyalari  : outputs/*_best.pth  (en iyi fold)")
    print(f"  - Arayuz icin      : python app.py")
    print_separator()


if __name__ == "__main__":
    main()
