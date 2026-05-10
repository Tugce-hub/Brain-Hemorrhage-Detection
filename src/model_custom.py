# -*- coding: utf-8 -*-
"""
model_custom.py — Model 2: MyCNN v2 — SE-Block Attention ile Guclendirilmis CNN.

v2 Degisiklikleri:
- SEBlock (Squeeze-and-Excitation): Kanal bazli attention mekanizmasi
- Double Conv per Block: Her blokta 2 ardisik conv → daha zengin feature
- Ayni genel mimari korundu (5 blok, GAP, FC head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── SE-Block (Squeeze-and-Excitation) ───────────────────────────────────────
class SEBlock(nn.Module):
    """
    Kanal bazli attention mekanizmasi.

    Squeeze: Global Average Pooling ile her kanalin oz-etini cikar
    Excitation: Kucuk FC ag ile kanal agirliklarini ogren
    Scale: Feature map'i kanal agirliklariyla carp

    Args:
        channels  (int): Giris kanal sayisi
        reduction (int): Daraltma orani (varsayilan 16 → C/16 noron)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        mid = max(channels // reduction, 4)   # minimum 4 noron
        self.squeeze    = nn.AdaptiveAvgPool2d(1)
        self.fc1        = nn.Linear(channels, mid, bias=False)
        self.fc2        = nn.Linear(mid, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Squeeze: (B, C, H, W) → (B, C)
        s = self.squeeze(x).view(b, c)
        # Excitation: (B, C) → (B, C/r) → (B, C)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        # Scale: kanalları ağırlıklandır
        return x * s


# ─── Gelismis ConvBlock (Double Conv + SE) ────────────────────────────────────
class ConvBlock(nn.Module):
    """
    Gelismis CNN bloku: (Conv→BN→ReLU) × 2 → SE-Block → Pool

    Double conv: Daha zengin feature cikartimi
    SE-Block:    Kanal bazli attention (hangi kanal onemli?)

    Args:
        in_channels  : Giris kanal sayisi
        out_channels : Cikis kanal sayisi
        pool         : MaxPooling uygulansin mi?
        use_se       : SE-Block kullanilsin mi?
        dilation     : Atrous conv (genisletme) carpani (varsayilan 1)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        pool:         bool = True,
        use_se:       bool = True,
        dilation:     int  = 1,
    ):
        super(ConvBlock, self).__init__()
        
        # Orijinal boyut koruma (same padding) dilation ile hesaplanir: p = d * (k-1) / 2
        pad = dilation

        # ── 1. Konvolusyon ────────────────────────────────────────────────────
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=pad, dilation=dilation, bias=False
        )
        self.bn1  = nn.BatchNorm2d(out_channels)

        # ── 2. Konvolusyon (Double Conv) ──────────────────────────────────────
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=pad, dilation=dilation, bias=False
        )
        self.bn2  = nn.BatchNorm2d(out_channels)

        # ── SE-Block ──────────────────────────────────────────────────────────
        self.se   = SEBlock(out_channels) if use_se else nn.Identity()

        # ── Pooling ───────────────────────────────────────────────────────────
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)   # 1. conv
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)   # 2. conv
        x = self.se(x)                                        # SE attention
        x = self.pool(x)                                      # pooling
        return x


# ─── MyCNN v2 ─────────────────────────────────────────────────────────────────
class MyCNN(nn.Module):
    """
    Ozgun sifirdan tasarlanmis CNN v2 (SE-Block + Double Conv).

    Mimari:
        Input (3 × 224 × 224)
        ↓ ConvBlock(3→32,   pool=True,  SE) → (32 × 112 × 112)
        ↓ ConvBlock(32→64,  pool=True,  SE) → (64 × 56 × 56)
        ↓ ConvBlock(64→128, pool=True,  SE) → (128 × 28 × 28)
        ↓ ConvBlock(128→256,pool=True,  SE) → (256 × 14 × 14)
        ↓ ConvBlock(256→512,pool=False, SE) → (512 × 14 × 14)
        ↓ Global Average Pooling            → (512,)
        ↓ FC(512→256) → BN → ReLU → Dropout
        ↓ FC(256→128) → BN → ReLU → Dropout
        ↓ FC(128→1)                         → (logit)

    Args:
        dropout_rate (float): FC katmanlarindaki Dropout orani.
    """

    def __init__(self, dropout_rate: float = 0.4):
        super(MyCNN, self).__init__()

        # ── Feature Extractor ─────────────────────────────────────────────────
        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True,  use_se=True),                      # 224→112
            ConvBlock(32,  64,  pool=True,  use_se=True),                      # 112→56
            ConvBlock(64,  128, pool=True,  use_se=True),                      # 56→28
            ConvBlock(128, 256, pool=True,  use_se=True, dilation=2),          # 28→14 (Görüntüleme alanini genişlet)
            nn.Dropout2d(p=0.1),                                               # Feature mapping spatial dropout
            ConvBlock(256, 512, pool=False, use_se=True, dilation=2),          # 14→14 (Geniş alan)
            nn.Dropout2d(p=0.2)                                                # Model ezberini zorla
        )

        # ── Global Pooling (Concat: Avg + Max) ────────────────────────────────
        self.gap_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.gap_max = nn.AdaptiveMaxPool2d(output_size=1)

        # ── Fully Connected Classifier ────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),  # Avg (512) + Max (512) = 1024
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.6),
            nn.Linear(128, 1),   # BCEWithLogitsLoss → sigmoid yok
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming He baslatmasi (ReLU aktivasyonu icin optimal)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)     # SE-Block + Dilated konvolusyon bloklari
        x_avg = self.gap_avg(x)
        x_max = self.gap_max(x)
        x = torch.cat((x_avg, x_max), dim=1)  # (B, 1024, 1, 1)
        x = self.classifier(x)   # Tam bagli katmanlar
        return x                  # logit shape: (B, 1)

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_mycnn(dropout_rate: float = 0.4) -> MyCNN:
    """MyCNN v4 ornegi olustur ve parametre bilgisini yazdir."""
    model = MyCNN(dropout_rate=dropout_rate)
    total = model.get_total_params()
    print(f"\n[Model] MyCNN v4 (ConcatPool + SE + Dilated)")
    print(f"   Toplam parametre: {total:,}")
    return model


# ─── Test ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_mycnn()
    dummy = torch.randn(4, 3, 224, 224)
    out   = model(dummy)
    print(f"\n   Input shape:  {dummy.shape}")
    print(f"   Output shape: {out.shape}")  # (4, 1)

    print("\nMimari Ozeti:")
    for name, module in model.named_children():
        print(f"   {name}: {module.__class__.__name__}")
