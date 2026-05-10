# 🧠 Beyin Kanaması Tespiti (Brain Hemorrhage Detection)

Bu proje, Bilgisayarlı Tomografi (CT) görüntülerinden beyin kanamasını otomatik olarak tespit etmek için geliştirilmiş bir Derin Öğrenme uygulamasıdır. Proje kapsamında hem özel tasarlanmış bir CNN (MyCNN) hem de önceden eğitilmiş ResNet50 modeli kullanılarak karşılaştırmalı analiz yapılmıştır.

## 🚀 Öne Çıkan Özellikler

- **Transfer Learning:** ResNet50 mimarisi kullanılarak yüksek doğruluk oranlarına ulaşılmıştır.
- **Custom CNN:** Parametre sayısı optimize edilmiş özel bir evrişimli sinir ağı (MyCNN) geliştirilmiştir.
- **Grid Search:** Hiperparametre optimizasyonu ile en iyi model konfigürasyonu belirlenmiştir.
- **K-Fold Cross Validation:** Modelin genelleme yeteneği 5-katlı çapraz doğrulama ile test edilmiştir.
- **Gradio Arayüzü:** Kullanıcıların görüntü yükleyerek anlık tahmin alabileceği web arayüzü.
- **Grad-CAM:** Modelin görüntünün hangi bölgesine odaklanarak karar verdiğini gösteren ısı haritaları.

## 📊 Model Performansı

Modellerin başarısı hem hiperparametre optimizasyonu (Grid Search) hem de genelleme yeteneğini test etmek için 5-Katlı Çapraz Doğrulama (K-Fold) ile değerlendirilmiştir.

### 1. Hiperparametre Optimizasyonu (Grid Search)
En iyi performansı veren parametre setleri:

| Model | Öğrenme Hızı | Dropout | Weight Decay | En İyi Val Acc (%) |
|:---|:---:|:---:|:---:|:---:|
| **ResNet50** | 0.001 | 0.3 | 0.0005 | **96.77** |
| **MyCNN** | 0.001 | 0.3 | 0.0002 | **90.32** |

### 2. 5-Katlı Çapraz Doğrulama (K-Fold CV) Sonuçları
Tüm veri seti üzerinde yapılan çapraz doğrulama sonuçları (Ortalama Değerler):

| Model | Ortalama Doğruluk (Accuracy) | Ortalama F1-Skoru |
|:---|:---:|:---:|
| **ResNet50** | **%97.00** | 0.969 |
| **MyCNN** | **%94.50** | 0.943 |

> **Not:** MyCNN modeli, k-katlı çapraz doğrulamada bazı katlarda %97.5'e kadar doğruluk göstererek ortalamada %94.5 başarıya ulaşmıştır.


## 🛠️ Kullanım

### 1. Eğitim (Training)
Tüm veri seti ile K-Fold çapraz doğrulama eğitimini başlatmak için:
```bash
python train_kfold.py
```

### 2. Hiperparametre Optimizasyonu (Grid Search)
```bash
python grid_search_resnet.py
python grid_search.py
```

### 3. Arayüzü Başlatma (Gradio)
Eğitilmiş modelleri kullanarak tahmin yapmak için:
```bash
python app.py
```

## 📁 Dosya Yapısı
- `src/`: Veri seti, model mimarileri ve eğitim lojiği.
- `head_ct/`: Görüntü veri seti.
- `outputs/`: Kaydedilen modeller (.pth), grafikler ve sonuçlar.
- `app.py`: Gradio web arayüzü uygulaması.
- `grid_search_resnet.py`: ResNet için optimizasyon betiği.

---
**Not:** Bu proje akademik amaçlarla geliştirilmiştir. Teşhis için profesyonel tıbbi yardım alınmalıdır.
