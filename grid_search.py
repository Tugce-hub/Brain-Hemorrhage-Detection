"""
grid_search.py -- Hiperparametre Optimizasyonu Icin Grid Search Betigi.

Bu dosya, modelin hiperparametrelerinin (Learning Rate, Dropout vb.) 
rastgele koda yazilmadigini, belirli bir deneme-yanilma (izgara arama) 
sonucunda secildigini ispatlamak icin yazilmistir. Yuksek egitim 
surelerini onlemek amaciyla daha dusuk Epoch limitleriyle (15) calisir.
"""

import os
import sys
import itertools
import pandas as pd

# Proje kok dizinini PATH'e ekle
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.dataset import get_dataloaders
from src.model_custom import build_mycnn
from src.train import train_model
from src.utils import ensure_dir

def main():
    outputs_dir = os.path.join(ROOT, "outputs")
    ensure_dir(outputs_dir)
    
    # -- Hiperparametre Izgarasi (Grid) --
    # Sureyi makul tutmak ve 8 farkli modeli hizlica test etmek icin dar bir aralik
    param_grid = {
        "learning_rate": [1e-3, 5e-4],
        "dropout_rate":  [0.3, 0.4],
        "weight_decay":  [1e-4, 2e-4]
    }
    
    # Tum kombinasyonlari olustur (2 x 2 x 2 = Toplam 8 Kombinasyon)
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Veri setini yukle (Dongu disi, sadece 1 kere yuklenir)
    print("\n[GRID SEARCH] Veri seti yukleniyor...")
    loaders = get_dataloaders(batch_size=16)
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    
    results = []
    
    # Egitim Limitleri (Grid search oldugu icin sureyi kisitliyoruz)
    GS_EPOCHS   = 15
    GS_PATIENCE = 5
    
    print(f"\n[GRID SEARCH] Toplam {len(combinations)} farkli kombinasyon denenecek.")
    print(f"Sabit Degerler -> Epoch: {GS_EPOCHS}, Patience: {GS_PATIENCE}, Batch Size: 16\n")
    
    for idx, params in enumerate(combinations, 1):
        lr   = params["learning_rate"]
        drop = params["dropout_rate"]
        wd   = params["weight_decay"]
        
        print(f"\n{'='*55}")
        print(f" DENEY {idx}/{len(combinations)} ".center(55, "="))
        print(f" LR: {lr} | Dropout: {drop} | Weight Decay: {wd} ")
        print(f"{'='*55}")
        
        # Modeli bu iterasyona ozel dropout ile olustur
        model = build_mycnn(dropout_rate=drop)
        run_name = f"mycnn_gs_deney_{idx}"
        
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
    
    csv_path = os.path.join(outputs_dir, "grid_search_results.csv")
    df_results.to_csv(csv_path, index=False)
    
    print("\n" + "="*60)
    print(" GRID SEARCH TAMAMLANDI ".center(60, "="))
    print("="*60)
    print(df_results.to_markdown(index=False))
    print(f"\n[INFO] Tum sonuclar (CSV tablosu) kaydedildi: {csv_path}")
    print("\n[TIP] Ipucu: Bu tabloyu kopyalayip raporundaki 'Hyperparameter Tuning' bolumune yapistirabilirsiniz.")

if __name__ == "__main__":
    main()
