# -*- coding: utf-8 -*-
"""
app.py - Beyin Kanamasi Tespiti Gradio Arayuzu (v2).

Ozellikler:
- CT goruntus yukleme
- ResNet50 ve MyCNN modelleri ile inference
- Hook tabanli Grad-CAM isi haritasi (dis kutuphane gerekmez)
- gr.Label ile dinamik progress bar'lar
- Karanlik premium tema, gradient buton hover efekti
- Bos durum (empty state) yonetimi
- Gradio 6.x uyumlu (css/theme -> launch())

Kullanim:
    python app.py
"""


import os
import sys

# ── Windows cp1254 encoding sorununu çöz: stdout/stderr'i UTF-8 yap ──────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import urllib.request
import json
import random
from io import BytesIO

import matplotlib
matplotlib.use("Agg")          # GUI backend olmadan çalıştır (sunucu ortamı)
import matplotlib.pyplot as plt

import gradio as gr

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.model_pretrained import build_resnet50
from src.model_custom     import build_mycnn
from src.transforms       import get_inference_transform


# ─── Sabitler ────────────────────────────────────────────────────────────────
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = {
    "ResNet50 (Transfer Learning)": os.path.join(OUTPUTS_DIR, "resnet50_best.pth"),
    "MyCNN (Özgün Tasarım)":        os.path.join(OUTPUTS_DIR, "mycnn_best.pth"),
    "Her İki Model (Karşılaştır)":  None,
}

CLASS_LABELS = {0: "✅ Normal", 1: "🩸 Kanamalı"}
CLASS_COLORS = {0: "#4CAF50",  1: "#F44336"}

transform = get_inference_transform()


# ─── Model Yükleme ───────────────────────────────────────────────────────────
def load_single_model(model_name: str, ckpt_path: str) -> nn.Module | None:
    """Checkpoint'ten model yükle ve eval moduna al."""
    if not os.path.exists(ckpt_path):
        return None

    if "ResNet50" in model_name or "resnet" in model_name.lower():
        model = build_resnet50(freeze_backbone=False, dropout_rate=0.5)
    else:
        model = build_mycnn(dropout_rate=0.5)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model


_model_cache: dict = {}


def get_model(display_name: str) -> nn.Module | None:
    """Model önbelleğinden al ya da yükle."""
    if display_name not in _model_cache:
        ckpt = MODEL_PATHS.get(display_name)
        if ckpt is None or not os.path.exists(ckpt):
            return None
        _model_cache[display_name] = load_single_model(display_name, ckpt)
    return _model_cache[display_name]


# ─── Grad-CAM ────────────────────────────────────────────────────────────────
def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Model mimarisine göre Grad-CAM için hedef konvolüsyon katmanını döndür.

    ResNet50:
        feature_extractor = Sequential(conv1, bn1, relu, maxpool,
                                        layer1, layer2, layer3, layer4, avgpool)
        → index -1 = avgpool,  index -2 = layer4  (hedef)

    MyCNN:
        features = Sequential(ConvBlock×5)
        → features[-1] = son ConvBlock  (hedef)
    """
    if "ResNet50" in model_name or "resnet" in model_name.lower():
        return model.feature_extractor[-2]   # layer4
    else:
        # features dizisinin sonu: ConvBlock (index -2) -> Dropout2d (index -1)
        # Grad-CAM için Dropout yerine ConvBlock'a bakmalıyız.
        return model.features[-2]            # son ConvBlock


def cam_to_overlay(cam: np.ndarray,
                   original_pil: Image.Image,
                   alpha: float = 0.45) -> Image.Image:
    """
    Normalize edilmiş CAM haritasını orijinal görüntü üzerine renkli overlay olarak uygula.

    Args:
        cam:          [H, W] numpy array, değerler [0, 1] aralığında
        original_pil: Orijinal PIL görüntüsü (herhangi boyut)
        alpha:        Isı haritasının yoğunluğu (0=sadece orijinal, 1=sadece heatmap)

    Returns:
        Blend edilmiş PIL görüntüsü
    """
    target_size = original_pil.size     # (W, H)

    # 1) CAM'i orijinal boyuta ölçekle
    cam_uint8   = (cam * 255).astype(np.uint8)
    cam_resized = np.array(
        Image.fromarray(cam_uint8).resize(target_size, Image.BILINEAR)
    )

    # 2) Jet renk haritası uygula
    cmap         = plt.cm.jet
    heatmap_rgba = cmap(cam_resized / 255.0)          # (H, W, 4)
    heatmap_rgb  = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    # 3) Orijinal görüntü ile harmanla (alpha blend)
    orig_arr = np.array(original_pil.convert("RGB")).astype(np.float32)
    heat_arr = heatmap_rgb.astype(np.float32)
    blended  = (alpha * heat_arr + (1.0 - alpha) * orig_arr).clip(0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def run_inference_with_gradcam(
    model:        nn.Module,
    model_name:   str,
    tensor:       torch.Tensor,
    original_pil: Image.Image,
) -> tuple:
    """
    Model üzerinde forward geçişi çalıştır, Grad-CAM hesapla ve overlay döndür.

    Grad-CAM adımları:
        1. Forward hook → son conv çıktısını (aktivasyonlar) yakala
        2. Backward hook → gradient'leri yakala
        3. Gradient'lerin global average pooling'i → kanal ağırlıkları
        4. Ağırlıklar × aktivasyonlar → ReLU → [0,1] normalize
        5. PIL overlay üret

    Returns:
        (pred_class_idx: int, probability: float, heatmap_pil: Image.Image)
    """
    _acts  = {}
    _grads = {}

    target_layer = get_target_layer(model, model_name)

    def _fwd_hook(module, inp, out):
        _acts["val"] = out

    def _bwd_hook(module, grad_in, grad_out):
        _grads["val"] = grad_out[0]

    h_fwd = target_layer.register_forward_hook(_fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(_bwd_hook)

    heatmap_pil = None
    prob = 0.5
    pred = 0

    try:
        model.zero_grad()
        logit = model(tensor)           # forward (gradient açık)
        logit.backward()                # backward (Grad-CAM için)

        acts  = _acts["val"].detach()   # (1, C, H, W)
        grads = _grads["val"].detach()  # (1, C, H, W)

        # Kanal ağırlıkları: her kanal için gradyanın uzamsal ortalaması
        weights = grads.mean(dim=(2, 3), keepdim=True)          # (1, C, 1, 1)
        cam     = F.relu((weights * acts).sum(dim=1))            # (1, H, W) — ReLU
        cam     = cam.squeeze(0).cpu().numpy()                   # (H, W)

        # [0, 1] aralığına normalize et
        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)

        prob = torch.sigmoid(logit).detach().item()
        pred = int(prob >= 0.5)

        heatmap_pil = cam_to_overlay(cam, original_pil)

    except Exception as exc:             # Grad-CAM başarısız olursa graceful fallback
        print(f"[!] Grad-CAM hatasi [{model_name}]: {exc}")
        heatmap_pil = original_pil.copy()
        # Normal inference ile tahmin al
        with torch.no_grad():
            logit = model(tensor)
            prob  = torch.sigmoid(logit).item()
            pred  = int(prob >= 0.5)

    finally:
        h_fwd.remove()
        h_bwd.remove()

    return pred, prob, heatmap_pil


# ─── Yer Tutucu Görüntü ──────────────────────────────────────────────────────
def _make_placeholder(width: int = 400, height: int = 300) -> Image.Image:
    """Başlangıç durumu için koyu gradyanlı yer tutucu PIL görüntüsü."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t   = y / height
        r   = int(14  + 8  * t)
        g   = int(14  + 8  * t)
        b   = int(28  + 20 * t)
        arr[y, :] = [r, g, b]
    return Image.fromarray(arr)


PLACEHOLDER_IMG = _make_placeholder()

# ─── Örnek Galeri (Veri Setinden Etiketli Görseller) ─────────────────────────
import pandas as pd
from sklearn.model_selection import train_test_split as _tts

_IMAGES_DIR = os.path.join(ROOT, "head_ct", "head_ct")
_LABELS_CSV = os.path.join(ROOT, "labels.csv")

# Test split'ten alınan bilinen etiketli görseller (eğitimde görmedi)
_GALLERY_ITEMS: list = []     # [(PIL, caption_str), ...]
_GALLERY_DF:    list = []     # [{"filename": ..., "label": ...}, ...]


def _build_gallery_df():
    """labels.csv'den test split'i yeniden oluştur (seed=42 sabit)."""
    df = pd.read_csv(_LABELS_CSV)
    df.columns  = df.columns.str.strip()
    df["filename"] = df["id"].apply(lambda x: f"{int(x):03d}.png")
    df["label"]    = df["hemorrhage"].astype(int)
    df = df[["filename", "label"]]

    # Aynı seed ile aynı split → test seti modelin görmediği veriler
    df_tv, df_test = _tts(df, test_size=0.15, stratify=df["label"], random_state=42)
    return df_test.reset_index(drop=True)


def _load_sample_gallery(n_per_class: int = 4):
    """
    Test setinden her siniftan rastgele n_per_class adet gorsel sec.
    Her cagrildiginda farkli gorseller dondurecek sekilde random.sample kullanir.
    gr.Gallery icin [(PIL, caption), ...] listesi dondurur.
    """
    import random as _rnd
    global _GALLERY_ITEMS, _GALLERY_DF
    _GALLERY_ITEMS = []
    _GALLERY_DF    = []

    try:
        df_test = _build_gallery_df()
        hemorrhage_rows = df_test[df_test["label"] == 1].to_dict("records")
        normal_rows     = df_test[df_test["label"] == 0].to_dict("records")

        # Her siniftan rastgele n_per_class adet sec
        picked_h = _rnd.sample(hemorrhage_rows, min(n_per_class, len(hemorrhage_rows)))
        picked_n = _rnd.sample(normal_rows,     min(n_per_class, len(normal_rows)))
        selected = picked_h + picked_n
        _rnd.shuffle(selected)   # karisik sirada goster

        for row in selected:
            path = os.path.join(_IMAGES_DIR, row["filename"])
            if not os.path.exists(path):
                continue
            pil     = Image.open(path).convert("RGB")
            label   = int(row["label"])
            caption = f"{'Kanamal\u0131' if label == 1 else 'Normal'}  ({row['filename']})"
            _GALLERY_ITEMS.append((pil, caption))
            _GALLERY_DF.append({"filename": row["filename"], "label": label})

    except Exception as e:
        print(f"[!] Galeri yukleme hatasi: {e}")

    return _GALLERY_ITEMS


def _gallery_select(evt: gr.SelectData):
    """
    Kullanıcı galeriden bir görsele tıkladığında:
      - Görseli image_input'a yükle
      - Gerçek etiketini HTML olarak göster (label KESIN bilinir)
    """
    idx = evt.index
    if idx >= len(_GALLERY_ITEMS):
        return None, ""

    pil   = _GALLERY_ITEMS[idx][0]
    info  = _GALLERY_DF[idx]
    label = info["label"]
    color = CLASS_COLORS[label]
    name  = CLASS_LABELS[label]

    html = (
        f"<div style='margin-top:8px; padding:10px 14px; background:#0A1A0A; "
        f"border:2px solid {color}; border-radius:9px; font-size:0.82em;'>"
        f"<b style='color:{color};'>✔ Gerçek Etiket: {name}</b><br>"
        f"<span style='color:#5050A0;'>Dosya: {info['filename']}</span><br>"
        f"<span style='color:#2A4A2A; font-size:0.78em;'>"
        f"labels.csv'den doğrulanmış — %100 güvenilir</span>"
        f"</div>"
    )
    return pil, html


# ─── Web Crawling (Geliştirilmiş — Etiketli Sorgular) ────────────────────────

# Sorgu → tahmini etiket eşlemesi
# Wikimedia Commons'ta bu kategorilerdeki görseller yüksek ihtimalle bu sınıftadır.
WEB_CRAWL_QUERIES = [
    # (sorgu_metni, tahmini_etiket, açıklama)
    ("brain hemorrhage CT scan",          1, "Kanamalı — genel beyin kanaması"),
    ("intracranial hemorrhage radiology",  1, "Kanamalı — intrakraniyal"),
    ("subdural hematoma CT",              1, "Kanamalı — subdural hematom"),
    ("epidural hematoma CT scan",         1, "Kanamalı — epidural hematom"),
    ("subarachnoid hemorrhage CT",        1, "Kanamalı — subaraknoid"),
    ("normal brain CT scan axial",        0, "Normal — aksiyel kesit"),
    ("healthy brain CT scan",             0, "Normal — sağlıklı beyin"),
    ("normal head CT without contrast",   0, "Normal — kontrastsız"),
]

_last_crawl_label: dict = {"label": None, "desc": ""}


def fetch_random_web_ct():
    """
    Wikimedia Commons API üzerinden etiketli sorgu havuzundan rastgele bir
    sorgu seçer, bulunan görseli indirir ve tahmini etiketi kaydeder.

    Etiket güvenilirliği:
      - Görseli TIP: sorgu terimi ile arama yapılıyor; etiket kesin değil,
        ama 'hemorrhage' içeren sorgular büyük olasılıkla kanamalı görsel verir.
      - Kullanıcıya gösterilen 'Tahmini Etiket' uyarısı ile belirtilir.

    Returns:
        PIL.Image veya None
    """
    import urllib.request, json, random
    from io import BytesIO

    custom_ua = "MedicalProjectCTFetcher/1.0 (Student University Project)"

    # Rastgele bir sorgu seç
    query_text, label_int, label_desc = random.choice(WEB_CRAWL_QUERIES)
    _last_crawl_label["label"] = label_int
    _last_crawl_label["desc"]  = label_desc

    print(f"[Web Crawler] Sorgu: '{query_text}'  →  Tahmini etiket: {label_desc}")
    # Wikimedia Commons API — birden fazla sayfa getir (gsrlimit=20)
    api_url = (
        "https://commons.wikimedia.org/w/api.php"
        "?action=query&format=json&prop=imageinfo"
        "&generator=search"
        f"&gsrsearch={query_text.replace(' ', '+')}"
        "&gsrnamespace=6"
        "&gsrlimit=20"
        "&iiprop=url|mime"
    )

    try:
        req  = urllib.request.Request(api_url, headers={"User-Agent": custom_ua})
        data = json.loads(urllib.request.urlopen(req, timeout=10).read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})

        urls = []
        for _, page_info in pages.items():
            for ii in page_info.get("imageinfo", []):
                img_url  = ii.get("url", "")
                mime     = ii.get("mime", "")
                if img_url.lower().endswith((".png", ".jpg", ".jpeg")) or "image" in mime:
                    urls.append(img_url)

        if not urls:
            print("[Web Crawler] Görsel bulunamadı.")
            return None

        # Rastgele bir URL seç ve indir
        selected = random.choice(urls)
        print(f"[Web Crawler] İndiriliyor: {selected}")
        img_req  = urllib.request.Request(selected, headers={"User-Agent": custom_ua})
        response = urllib.request.urlopen(img_req, timeout=15)
        return Image.open(BytesIO(response.read())).convert("RGB")

    except Exception as e:
        print(f"[!] Web crawling hatası: {e}")
        return None


def get_last_crawl_label_html() -> str:
    """Son web crawl işleminin tahmini etiketini HTML olarak döndür."""
    label = _last_crawl_label.get("label")
    desc  = _last_crawl_label.get("desc", "")
    if label is None:
        return ""
    color = CLASS_COLORS[label]
    name  = CLASS_LABELS[label]
    return (
        f"<div style='margin-top:8px; padding:10px 14px; background:#12122A; "
        f"border:1px solid {color}44; border-radius:9px; font-size:0.82em;'>"
        f"<b style='color:{color};'>Tahmini Etiket: {name}</b><br>"
        f"<span style='color:#5050A0;'>{desc}</span><br>"
        f"<span style='color:#383858; font-size:0.78em;'>⚠ Wikimedia sorgusu baz alınmıştır, kesin değildir.</span>"
        f"</div>"
    )




# ─── Gradio Tahmin Fonksiyonu ─────────────────────────────────────────────────
def predict(pil_image, model_choice: str) -> tuple:
    """
    Gradio'nun çağırdığı ana tahmin fonksiyonu.

    Returns:
        (original_pil, heatmap_pil, label_dict, info_html)
        - original_pil : gr.Image
        - heatmap_pil  : gr.Image  (Grad-CAM overlay)
        - label_dict   : gr.Label  {"🩸 Kanamalı": prob, "✅ Normal": 1-prob}
        - info_html    : gr.HTML   (sonuç kartı veya karşılaştırma paneli)
    """
    if pil_image is None:
        return (
            PLACEHOLDER_IMG,
            PLACEHOLDER_IMG,
            None,
            _empty_state_html("⚠️ Lütfen bir görüntü yükleyin."),
        )

    image_rgb = pil_image.convert("RGB")
    tensor    = transform(image_rgb).unsqueeze(0).to(DEVICE)

    # ── Karşılaştırma modu ────────────────────────────────────────────────
    if model_choice == "Her İki Model (Karşılaştır)":
        models_info = [
            "ResNet50 (Transfer Learning)",
            "MyCNN (Özgün Tasarım)",
        ]

        results    = []
        first_hmap = None
        html_parts = []

        for mname in models_info:
            model = get_model(mname)
            if model is None:
                html_parts.append(
                    f"<div style='padding:12px; background:#1A1A2E; border:1px solid #EF9A9A;"
                    f"border-radius:10px; margin:6px 0;'>"
                    f"<b style='color:#EF9A9A'>⚠️ {mname}</b><br>"
                    f"<span style='color:#7070A0; font-size:0.85em;'>"
                    f"Model bulunamadı. Önce: <code>python train_kfold.py</code></span>"
                    f"</div>"
                )
                continue

            pred, prob, hmap = run_inference_with_gradcam(model, mname, tensor, image_rgb)

            if first_hmap is None:
                first_hmap = hmap

            results.append((mname, pred, prob))

            label      = CLASS_LABELS[pred]
            color      = CLASS_COLORS[pred]
            pct        = prob * 100 if pred == 1 else (1 - prob) * 100
            bar_kanama = prob * 100
            bar_normal = (1 - prob) * 100

            html_parts.append(
                f"<div style='padding:16px; background:#1A1A2E; border:2px solid {color};"
                f"border-radius:14px; margin:8px 0;'>"
                f"<b style='font-size:0.9em; color:#9090C0; letter-spacing:0.5px;'>{mname}</b>"
                f"<div style='font-size:2em; font-weight:900; color:{color};"
                f"text-shadow:0 0 16px {color}55; margin:6px 0;'>{label}</div>"
                f"<div style='color:#888; font-size:0.82em; margin-bottom:10px;'>"
                f"Güven skoru: <b style='color:white;'>{pct:.1f}%</b></div>"
                # Kanamalı bar
                f"<div style='font-size:0.78em; color:#888; margin-bottom:3px;'>🩸 Kanamalı</div>"
                f"<div style='background:#12122A; border-radius:6px; height:10px; overflow:hidden; margin-bottom:7px;'>"
                f"<div style='width:{bar_kanama:.1f}%; background:linear-gradient(90deg,#F44336,#FF7043);"
                f"height:100%; border-radius:6px;'></div></div>"
                # Normal bar
                f"<div style='font-size:0.78em; color:#888; margin-bottom:3px;'>✅ Normal</div>"
                f"<div style='background:#12122A; border-radius:6px; height:10px; overflow:hidden;'>"
                f"<div style='width:{bar_normal:.1f}%; background:linear-gradient(90deg,#4CAF50,#8BC34A);"
                f"height:100%; border-radius:6px;'></div></div>"
                f"</div>"
            )

        # gr.Label: ilk modelin tahminini kullan
        if results:
            _, _, p0 = results[0]
            label_dict = {"🩸 Kanamalı": round(p0, 4), "✅ Normal": round(1 - p0, 4)}
        else:
            label_dict = {"🩸 Kanamalı": 0.5, "✅ Normal": 0.5}

        compare_html = (
            "<div style='background:#0E0E1E; border-radius:14px; padding:4px;'>"
            + "".join(html_parts)
            + "</div>"
        )

        return image_rgb, first_hmap or PLACEHOLDER_IMG, label_dict, compare_html

    # ── Tek model modu ────────────────────────────────────────────────────
    model = get_model(model_choice)
    if model is None:
        return (
            image_rgb,
            PLACEHOLDER_IMG,
            None,
            "<p style='color:#EF9A9A; padding:20px;'>⚠️ Model dosyası bulunamadı. "
            "Önce <code>python train_kfold.py</code> ile eğitimi tamamlayın.</p>",
        )

    pred, prob, heatmap_pil = run_inference_with_gradcam(model, model_choice, tensor, image_rgb)

    label      = CLASS_LABELS[pred]
    color      = CLASS_COLORS[pred]
    pct        = prob * 100 if pred == 1 else (1 - prob) * 100
    label_dict = {"🩸 Kanamalı": round(prob, 4), "✅ Normal": round(1 - prob, 4)}
    bar_kanama = prob * 100
    bar_normal = (1 - prob) * 100

    info_html = (
        f"<div style='padding:20px; background:#1A1A2E; border:3px solid {color};"
        f"border-radius:16px; text-align:center;'>"
        # Ana tahmin etiketi
        f"<div style='font-size:2.6em; font-weight:900; color:{color};"
        f"text-shadow:0 0 24px {color}55; letter-spacing:1px;'>{label}</div>"
        f"<div style='font-size:1em; color:#9090B0; margin-top:8px;'>"
        f"Güven Skoru: <b style='color:white; font-size:1.3em;'>{pct:.1f}%</b></div>"
        f"<div style='color:#404060; font-size:0.8em; margin-top:4px;'>"
        f"Ham olasılık (Kanamalı): {prob*100:.2f}%</div>"
        f"<hr style='border:none; border-top:1px solid #2A2A45; margin:14px 0;'>"
        # Progress bars
        f"<div style='text-align:left;'>"
        f"<div style='font-size:0.82em; color:#888; margin-bottom:3px;'>🩸 Kanamalı</div>"
        f"<div style='background:#12122A; border-radius:8px; height:14px; overflow:hidden; margin-bottom:9px;'>"
        f"<div style='width:{bar_kanama:.1f}%; background:linear-gradient(90deg,#F44336,#FF7043);"
        f"height:100%; border-radius:8px; transition:width 0.6s ease;'></div></div>"
        f"<div style='font-size:0.82em; color:#888; margin-bottom:3px;'>✅ Normal</div>"
        f"<div style='background:#12122A; border-radius:8px; height:14px; overflow:hidden;'>"
        f"<div style='width:{bar_normal:.1f}%; background:linear-gradient(90deg,#4CAF50,#8BC34A);"
        f"height:100%; border-radius:8px; transition:width 0.6s ease;'></div></div>"
        f"</div>"
        f"</div>"
    )

    return image_rgb, heatmap_pil, label_dict, info_html


# ─── Yardımcı HTML ───────────────────────────────────────────────────────────
def _empty_state_html(msg: str = "") -> str:
    return (
        "<div style='text-align:center; color:#404060; padding:28px;"
        "background:#0E0E1E; border-radius:14px; border:1px dashed #2A2A45;'>"
        "<div style='font-size:2.4em; margin-bottom:10px;'>🧠</div>"
        f"<div style='font-size:0.9em; color:#6060A0;'>{msg or 'Görüntü yükleyip'} "
        "<b style='color:#8B5CF6;'>Analiz Et</b> butonuna tıklayın</div>"
        "</div>"
    )


# ─── CSS ─────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Genel Gövde ─────────────────────────────────────────── */
body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Analiz Et Butonu ────────────────────────────────────── */
#analyze_button {
    background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    letter-spacing: 0.4px !important;
    border-radius: 10px !important;
    transition: transform 0.2s ease, box-shadow 0.25s ease !important;
}
#analyze_button:hover {
    transform: scale(1.04) !important;
    box-shadow: 0 0 28px rgba(139, 92, 246, 0.65), 0 4px 16px rgba(59, 130, 246, 0.4) !important;
}
#analyze_button:active {
    transform: scale(0.98) !important;
}

#crawler_button {
    margin-bottom: 15px !important;
    border: 1px solid #2A2A45 !important;
    background: #12122A !important;
    color: #9090C0 !important;
    transition: all 0.2s ease !important;
}
#crawler_button:hover {
    background: #1A1A2E !important;
    border-color: #8B5CF6 !important;
    color: white !important;
}

#gallery_refresh_button {
    width: 100% !important;
    border: 1px dashed #2A3A2A !important;
    background: #0E1A0E !important;
    color: #60A060 !important;
    font-size: 0.82em !important;
    transition: all 0.2s ease !important;
    margin-top: 4px !important;
    margin-bottom: 10px !important;
}
#gallery_refresh_button:hover {
    background: #122012 !important;
    border-color: #4CAF50 !important;
    color: #90EE90 !important;
    transform: scale(1.01) !important;
}

/* ── Başlık Alanı ────────────────────────────────────────── */
.title-text {
    text-align: center;
    font-size: 2.4em;
    font-weight: 800;
    background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 50%, #F093FB 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding: 20px 0 6px;
    letter-spacing: -0.5px;
}
.subtitle-text {
    text-align: center;
    color: #6060A0;
    font-size: 0.95em;
    margin-bottom: 18px;
    line-height: 1.6;
}

/* ── Görüntü Bileşenleri ─────────────────────────────────── */
#output_original img,
#output_heatmap img {
    border-radius: 10px;
    border: 1px solid #2A2A45;
}

/* ── Label (progress bar) bileşeni ──────────────────────── */
#output_label .label-container {
    background: #12122A !important;
    border-radius: 10px !important;
    border: 1px solid #2A2A45 !important;
}
"""


# ─── Arayüz Tanımı ───────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Beyin Kanaması Tespiti") as demo:

        # ── Başlık ──────────────────────────────────────────────────────────
        gr.HTML("""
            <div class='title-text'>🧠 Beyin Kanaması Tespiti</div>
            <div class='subtitle-text'>
                Head CT görüntüsü yükleyin — Yapay zeka kanamalı / normal olduğunu tahmin etsin<br>
                <span style='font-size:0.85em; color:#44446A;'>
                    ✨ Grad-CAM ile açıklanabilir AI &nbsp;|&nbsp; 🔬 ResNet50 & MyCNN
                </span>
            </div>
        """)

        gr.HTML("<hr style='border:none; border-top:1px solid #2A2A45; margin:0 0 18px;'>")

        with gr.Row(equal_height=False):

            # ── Sol: Giriş Paneli ────────────────────────────────────────────
            with gr.Column(scale=1, min_width=260):

                gr.HTML(
                    "<div style='color:#7070B0; font-size:0.78em; font-weight:600;"
                    "text-transform:uppercase; letter-spacing:1.2px; margin-bottom:8px;'>"
                    "📂 Giriş Paneli</div>"
                )

                image_input = gr.Image(
                    type="pil",
                    label="CT Görüntüsü Yükle",
                    height=280,
                    elem_id="ct_image_input",
                )

            # ─── Örnek Galeri Paneli ─────────────────────────────────────
                gr.HTML(
                    "<div style='color:#7070B0; font-size:0.78em; font-weight:600;"
                    "text-transform:uppercase; letter-spacing:1.2px;"
                    "margin:16px 0 8px;'>"
                    "🖼️ Veri Setinden Örnek Görseller</div>"
                )

                sample_gallery = gr.Gallery(
                    value=_load_sample_gallery(),
                    label="Tikla → Yukle & Analiz Et",
                    columns=4,
                    rows=2,
                    height=220,
                    elem_id="sample_gallery",
                    allow_preview=False,
                )

                gallery_refresh_btn = gr.Button(
                    "\U0001f504 Farkli Gorsel Goster",
                    variant="secondary",
                    size="sm",
                    elem_id="gallery_refresh_button",
                )

                crawl_label_html = gr.HTML("", elem_id="crawl_label_html")

                random_web_btn = gr.Button(
                    "🌐 Web'den Rastgele CT Çek (Crawler)",
                    variant="secondary",
                    elem_id="crawler_button"
                )

                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_PATHS.keys()),
                    value="Her İki Model (Karşılaştır)",
                    label="🤖 Model Seçimi",
                    elem_id="model_selector",
                )

                submit_btn = gr.Button(
                    "🔍  Analiz Et",
                    variant="primary",
                    size="lg",
                    elem_id="analyze_button",
                )

                gr.HTML("""
                <div style='margin-top:14px; padding:13px; background:#0E0E1E;
                     border-radius:11px; border:1px solid #2A2A44;'>
                    <div style='color:#6060A0; font-size:0.78em; line-height:1.7;'>
                        <b style='color:#8080B0;'>ℹ️ Nasıl çalışır?</b><br>
                        Model CT görüntüsünü analiz eder ve<br>
                        <b style='color:#8B5CF6;'>Grad-CAM</b> ile odaklandığı<br>
                        bölgeyi ısı haritası olarak gösterir.
                    </div>
                </div>
                """)

            # ── Sağ: Sonuç Paneli ────────────────────────────────────────────
            with gr.Column(scale=2):

                gr.HTML(
                    "<div style='color:#7070B0; font-size:0.78em; font-weight:600;"
                    "text-transform:uppercase; letter-spacing:1.2px; margin-bottom:8px;'>"
                    "📊 Analiz Sonucu</div>"
                )

                with gr.Row(equal_height=True):
                    output_original = gr.Image(
                        value=PLACEHOLDER_IMG,
                        label="🔬 Orijinal CT",
                        height=270,
                        elem_id="output_original",
                        interactive=False,
                    )
                    output_heatmap = gr.Image(
                        value=PLACEHOLDER_IMG,
                        label="🌡️ Grad-CAM Isi Haritasi",
                        height=270,
                        elem_id="output_heatmap",
                        interactive=False,
                    )

                output_label = gr.Label(
                    value=None,
                    label="📈 Tahmin Olasılıkları",
                    num_top_classes=2,
                    elem_id="output_label",
                )

                output_html = gr.HTML(
                    value=_empty_state_html(),
                    elem_id="output_html",
                )

        # ── Buton Aksiyonları ──────────────────────────────────────────────────
        def _on_crawl():
            img = fetch_random_web_ct()
            html = get_last_crawl_label_html()
            return img, html

        random_web_btn.click(
            fn=_on_crawl,
            inputs=[],
            outputs=[image_input, crawl_label_html],
        )

        # Galeriden gorsel secilince image_input'a yukle
        sample_gallery.select(
            fn=_gallery_select,
            inputs=[],
            outputs=[image_input, crawl_label_html],
        )

        # Galeri yenile butonu — rastgele yeni 8 gorsel yukle
        gallery_refresh_btn.click(
            fn=_load_sample_gallery,
            inputs=[],
            outputs=[sample_gallery],
        )

        submit_btn.click(
            fn=predict,
            inputs=[image_input, model_dropdown],
            outputs=[output_original, output_heatmap, output_label, output_html],
        )

        # ── Bilgi Kartları ───────────────────────────────────────────────────
        gr.HTML("<hr style='border:none; border-top:1px solid #2A2A45; margin:20px 0 16px;'>")

        with gr.Row():
            gr.Markdown("""
### 📖 Nasıl Kullanılır?
1. Sol taraftan bir **Head CT görüntüsü** yükleyin
2. Kullanmak istediğiniz **modeli** seçin
3. **"Analiz Et"** butonuna tıklayın
4. Tahmin + **Grad-CAM ısı haritasını** görün
            """)
            gr.Markdown("""
### 🏗️ Modeller
| Model | Tür | Özellik |
|---|---|---|
| **ResNet50** | Transfer Learning | ImageNet önceden eğitilmiş |
| **MyCNN** | Özgün Tasarım | Sıfırdan yazılmış CNN |
| **Her İkisi** | Karşılaştırma | İki model yan yana |
            """)
            gr.Markdown("""
### ⚠️ Önemli Not
Bu uygulama bir **üniversite projesidir** ve yalnızca eğitim amaçlıdır.
Tanı için **tıbbi bir uzmana** başvurunuz.

**Sınıflar:**
- 🩸 **Kanamalı** (hemorrhage = 1)
- ✅ **Normal** (hemorrhage = 0)
            """)

    return demo


# ─── Ana ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[*] Gradio arayuzu (v2) baslatiliyor...")
    print(f"   Cihaz       : {DEVICE}")
    print(f"   Model dizini: {OUTPUTS_DIR}")

    for name, path in MODEL_PATHS.items():
        if path is not None:
            status = "[OK]" if (path and os.path.exists(path)) else "[!] (Egitim gerekli)"
            print(f"   {status} {name}")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        show_error=True,
        inbrowser=True,
        # ── Gradio 6.x: theme ve css buraya taşındı ──
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="purple",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif"],
        ),
        css=CSS,
    )
