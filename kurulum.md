# 🛠️ Kurulum Rehberi

Bu proje Python 3.10+ ve PyTorch kullanılarak geliştirilmiştir. Aşağıdaki adımları izleyerek projeyi kendi ortamınızda çalıştırabilirsiniz.

## 1. Gereksinimler

Proje için gerekli kütüphaneler `requirements.txt` dosyasında listelenmiştir. Temel bağımlılıklar:
- PyTorch & Torchvision
- Pandas & Numpy
- Scikit-learn
- Matplotlib
- Gradio (Arayüz için)
- Tqdm (Progress bar için)

## 2. Kurulum Adımları

### Adım 1: Depoyu Klonlayın veya İndirin
Projeyi bilgisayarınıza indirdikten sonra terminal üzerinden proje klasörüne gidin.

### Adım 2: Sanal Ortam Oluşturun (Önerilen)
```bash
python -m venv venv
# Windows için:
venv\Scripts\activate
# Linux/Mac için:
source venv/bin/activate
```

### Adım 3: Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

## 3. Veri Seti Hazırlığı

Veri seti `head_ct` klasörü altında bulunmalıdır:
- `head_ct/head_ct/`: Görüntü dosyaları (.png)
- `labels.csv`: Görüntü isimlerini ve etiketleri (0/1) içeren dosya.

## 4. Modelleri Çalıştırma

### Modeli Eğitmek:
```bash
python train_kfold.py
```
Bu komut hem **MyCNN** hem de **ResNet50** modellerini eğitecek şekilde yapılandırılabilir ve en iyi ağırlıkları `outputs/` klasörüne kaydeder.

### Test ve Değerlendirme:
```bash
python run_eval.py
```

### Kullanıcı Arayüzünü Başlatmak:
```bash
python app.py
```

## ⚠️ Önemli Notlar
- **CUDA Desteği:** Eğer NVIDIA GPU'nuz varsa, PyTorch otomatik olarak GPU'yu kullanacaktır. GPU kullanımı eğitim süresini önemli ölçüde kısaltır.
- **Bellek Hataları:** Eğer bellek (RAM/VRAM) hatası alırsanız, `train_kfold.py` içerisindeki `batch_size` değerini düşürebilirsiniz.
