"""
model_pretrained.py — Model 1: ResNet50 ile Transfer Learning.

Strateji:
1. ImageNet ön-eğitimli ResNet50 yükle
2. Tüm katmanları dondur (frozen)
3. Son fc katmanını ikili sınıflandırma için değiştir
4. Sadece son katmanı (ve opsiyonel olarak layer3/layer4) eğit
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class ResNet50Classifier(nn.Module):
    """
    Transfer Learning ile ResNet50 tabanlı ikili sınıflandırıcı.

    Args:
        freeze_backbone (bool): True → feature extractor katmanlarını dondur,
                                 sadece classifier ve son layer'ları eğit.
        dropout_rate (float):   Classifier'daki Dropout oranı.
    """

    def __init__(self, freeze_backbone: bool = True, dropout_rate: float = 0.5):
        super(ResNet50Classifier, self).__init__()

        # ── ImageNet ağırlıklarıyla ResNet50 yükle ───────────────────────────
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # ── Backbone'u dondur ─────────────────────────────────────────────────
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # ── layer4'ü unfreeze et (ince ayar / fine-tuning) ───────────────────
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        # ── Mevcut fc'yi al ve kaldır ─────────────────────────────────────────
        in_features = backbone.fc.in_features  # 2048

        # Backbone'dan fc'yi çıkar (feature extractor olarak kullan)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # ── Özel Sınıflandırıcı ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.6),
            nn.Linear(256, 1)   # BCEWithLogitsLoss → sigmoid yok
        )

        # Classifier ağırlıklarını Xavier ile başlat
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)  # (B, 2048, 1, 1)
        logits   = self.classifier(features)   # (B, 1)
        return logits

    def unfreeze_all(self):
        """Tam fine-tuning icin tum agirliklari serbest birak."""
        for param in self.parameters():
            param.requires_grad = True
        print("  [i] Tum katmanlar serbest birakildi (full fine-tuning modu).")

    def get_trainable_params(self) -> int:
        """Eğitilebilir parametre sayısını döndür."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Toplam parametre sayısını döndür."""
        return sum(p.numel() for p in self.parameters())


def build_resnet50(freeze_backbone: bool = True, dropout_rate: float = 0.5) -> ResNet50Classifier:
    """ResNet50Classifier ornek olustur ve parametre bilgisini yazdir."""
    model = ResNet50Classifier(
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate
    )
    trainable = model.get_trainable_params()
    total     = model.get_total_params()
    print(f"\n[Model] ResNet50 (Transfer Learning)")
    print(f"   Toplam parametre:      {total:,}")
    print(f"   Egitilebilir:          {trainable:,}  ({100*trainable/total:.1f}%)")
    return model


# ─── Test ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_resnet50(freeze_backbone=True)
    dummy = torch.randn(4, 3, 224, 224)
    out   = model(dummy)
    print(f"\n   Input shape:  {dummy.shape}")
    print(f"   Output shape: {out.shape}")   # (4, 1)
