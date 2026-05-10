"""
grid_search_resnet.py -- ResNet50 icin Hiperparametre Optimizasyonu.

Bu dosya, ResNet50 modelinin hiperparametrelerinin (Learning Rate, Dropout vb.) 
belirli bir deneme-yanilma (izgara arama) sonucunda secildigini ispatlamak icin yazilmistir. 
Sureyi makul tutmak icin 15 epoch ile calisir.
"""

import os
import sys
import itertools
import pandas as pd

# Proje kok dizinini PATH'e ekle
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.dataset import get_dataloaders
from src.model_pretrained import build_resnet50
from src.train import train_model
from src.utils import ensure_dir

def main():
    outputs_dir = os.path.join(ROOT, "outputs")
    ensure_dir(outputs_dir)
    
    # -- Hiperparametre Izgarasi (Grid) --
    # 2 x 2 x 2 = 8 Farkli Kombinasyon
    param_grid = {
        "learning_rate": [1e-3, 1e-4],
        "dropout_rate":  [0.3, 0.5],
        "weight_decay":  [1e-4, 5e-4]
    }
    
    # Tum kombinasyonlari olustur
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Veri setini yukle
    print("\n[GRID SEARCH - RESNET] Veri seti yukleniyor...")
    loaders = get_dataloaders(batch_size=16)
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    
    results = []
    GS_EPOCHS   = 15
    GS_PATIENCE = 5
    
    print(f"\n[GRID SEARCH] Toplam {len(combinations)} farkli kombinasyon denenecek.")
    print(f"Sabit Degerler -> Epoch: {GS_EPOCHS}, Patience: {GS_PATIENCE}, Batch Size: 16\n")
    
    for idx, params in enumerate(combinations, 1):
        lr   = params["learning_rate"]
        drop = params["dropout_rate"]
        wd   = params["weight_decay"]
        
        print(f"\n{'='*55}")
        print(f" RESNET DENEY {idx}/{len(combinations)} ".center(55, "="))
        print(f" LR: {lr} | Dropout: {drop} | Weight Decay: {wd} ")
        print(f"{'='*55}")
        
        # Modeli bu iterasyona ozel dropout ile olustur
        model = build_resnet50(dropout_rate=drop, freeze_backbone=True)
        run_name = f"resnet50_gs_deney_{idx}"
        
        # Egitimi baslat
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=run_name,
            save_dir=outputs_dir,
            num_epochs=GS_EPOCHS,
            learning_rate=lr,
            weight_decay=wd,
            patience=GS_PATIENCE,
            use_amp=True,
            scheduler_type="cosine"
        )
        
        best_val_loss = history["best_val_loss"]
        best_epoch_idx = history["best_epoch"] - 1
        
        # Ilgili epoch'taki Validation dogruluk orani
        try:
            best_val_acc = history["val_acc"][best_epoch_idx]
        except IndexError:
            best_val_acc = 0.0
        
        # Sonucu sozluğe kaydet
        results.append({
            "Deney_No": idx,
            "Learning_Rate": lr,
            "Dropout": drop,
            "Weight_Decay": wd,
            "Best_Epoch": history["best_epoch"],
            "Best_Val_Loss": round(best_val_loss, 4),
            "Best_Val_Acc(%)": round(best_val_acc * 100, 2)
        })
        
    # Pandas ile tabloyu olustur ve CSV'ye yaz
    df_results = pd.DataFrame(results)
    
    # En iyi Doğruluk Oranina gore azalan sekilde sirala
    df_results = df_results.sort_values(by="Best_Val_Acc(%)", ascending=False)
    
    csv_path = os.path.join(outputs_dir, "grid_search_resnet_results.csv")
    df_results.to_csv(csv_path, index=False)
    
    print("\n" + "="*60)
    print(" RESNET GRID SEARCH TAMAMLANDI ".center(60, "="))
    print("="*60)
    print(df_results.to_markdown(index=False))
    print(f"\n[INFO] Tum sonuclar (CSV tablosu) kaydedildi: {csv_path}")

if __name__ == "__main__":
    main()
