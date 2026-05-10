"""
run_eval.py -- Degerlendirme scriptini calistir, sonucu dosyaya yaz.
"""
import os, sys, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

log_path = os.path.join("outputs", "eval_result.txt")

with open(log_path, "w", encoding="utf-8") as logf:
    def log(msg=""):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    try:
        import torch
        from src.dataset import get_dataloaders
        from src.evaluate import evaluate_model, print_comparison_table, OUTPUTS_DIR
        from src.utils import print_separator

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Cihaz: {device}")

        loaders = get_dataloaders(batch_size=16)
        test_loader = loaders["test"]

        results = []
        for name in ["resnet50", "mycnn"]:
            try:
                r = evaluate_model(name, test_loader, device, save_dir=OUTPUTS_DIR)
                results.append(r)
            except Exception as e:
                log(f"[HATA] {name}: {e}")
                traceback.print_exc()

        if len(results) > 1:
            print_comparison_table(results)

        log("\n=== MODEL KARSILASTIRMA ===")
        log(f"{'Model':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        log("-" * 55)
        for r in results:
            log(f"{r['model']:<18} {r['accuracy']*100:>9.2f}% {r['precision']*100:>9.2f}% {r['recall']*100:>9.2f}% {r['f1']*100:>9.2f}%")
        log("-" * 55)

        best = max(results, key=lambda x: x["f1"])
        log(f"\nEn iyi model (F1): {best['model']} | F1={best['f1']*100:.2f}% | Acc={best['accuracy']*100:.2f}%")
        log("\n[TAMAM] Degerlendirme tamamlandi!")
        log(f"Sonuc dosyasi: {os.path.abspath(log_path)}")

    except Exception as e:
        log(f"\n[KRITIK HATA] {e}")
        traceback.print_exc()
