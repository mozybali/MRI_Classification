#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model_comparison.py
-------------------
Birden fazla modeli karşılaştırma ve en iyi model seçimi scripti.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import numpy as np

from ayarlar import MODELS_KLASORU, GORSELLER_KLASORU


def model_listesi_al() -> List[Path]:
    """Tüm eğitilmiş modelleri listele."""
    modeller = list(MODELS_KLASORU.glob("*.pkl"))
    return sorted(modeller, key=lambda p: p.stat().st_mtime, reverse=True)


def model_metriklerini_oku(model_yolu: Path) -> Dict:
    """Model metadata'sından metrikleri oku."""
    metadata_yolu = model_yolu.with_suffix('.json')
    
    if not metadata_yolu.exists():
        return None
    
    with open(metadata_yolu, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return metadata


def karsilastirma_tablosu_olustur(modeller: List[Path]) -> pd.DataFrame:
    """Model karşılaştırma tablosu oluştur."""
    veriler = []
    
    for model_yolu in modeller:
        metadata = model_metriklerini_oku(model_yolu)
        
        if metadata is None:
            continue
        
        metrikler = metadata.get('metrikler', {})
        
        veri = {
            'Model': model_yolu.stem,
            'Tip': metadata.get('model_tipi', 'N/A').upper(),
            'Tarih': metadata.get('tarih', 'N/A')[:10],
            'Accuracy': metrikler.get('accuracy', 0),
            'Precision': metrikler.get('precision_macro', 0),
            'Recall': metrikler.get('recall_macro', 0),
            'F1-Score': metrikler.get('f1_macro', 0),
            'Cohen Kappa': metrikler.get('cohen_kappa', 0),
            'ROC-AUC': metrikler.get('roc_auc_ovr', 0) if metrikler.get('roc_auc_ovr') else 0
        }
        
        veriler.append(veri)
    
    if not veriler:
        return None
    
    df = pd.DataFrame(veriler)
    return df


def en_iyi_model_sec(df: pd.DataFrame, metrik: str = 'F1-Score') -> str:
    """Belirtilen metriğe göre en iyi modeli seç."""
    if df is None or df.empty:
        return None
    
    en_iyi_idx = df[metrik].idxmax()
    return df.loc[en_iyi_idx, 'Model']


def karsilastirma_grafigi_ciz(df: pd.DataFrame):
    """Model karşılaştırma grafiği çiz."""
    if df is None or df.empty:
        print("⚠️  Grafik çizilemedi: Veri yok")
        return
    
    # Metrikler
    metrikler = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Cohen Kappa']
    metrikler_mevcut = [m for m in metrikler if m in df.columns]
    
    if not metrikler_mevcut:
        print("⚠️  Çizilebilecek metrik bulunamadı")
        return
    
    # Grafik
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metrik in enumerate(metrikler_mevcut):
        ax = axes[idx]
        
        # Bar plot
        df_sorted = df.sort_values(metrik, ascending=False)
        bars = ax.bar(range(len(df_sorted)), df_sorted[metrik], 
                     color=['green' if i == 0 else 'steelblue' for i in range(len(df_sorted))])
        
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels([f"{row['Tip']}\n{row['Tarih']}" 
                           for _, row in df_sorted.iterrows()], 
                          rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(metrik, fontsize=11)
        ax.set_title(f'{metrik} Karşılaştırması', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Değerleri bar'ların üstüne yaz
        for i, (bar, val) in enumerate(zip(bars, df_sorted[metrik])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # ROC-AUC (varsa)
    if 'ROC-AUC' in df.columns and idx < len(axes) - 1:
        idx += 1
        ax = axes[idx]
        df_sorted = df.sort_values('ROC-AUC', ascending=False)
        bars = ax.bar(range(len(df_sorted)), df_sorted['ROC-AUC'],
                     color=['green' if i == 0 else 'orange' for i in range(len(df_sorted))])
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels([f"{row['Tip']}\n{row['Tarih']}" 
                           for _, row in df_sorted.iterrows()],
                          rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('ROC-AUC', fontsize=11)
        ax.set_title('ROC-AUC Karşılaştırması', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, df_sorted['ROC-AUC'])):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Boş subplot'ları gizle
    for idx in range(len(metrikler_mevcut) + (1 if 'ROC-AUC' in df.columns else 0), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Kaydet
    cikti_yolu = GORSELLER_KLASORU / "model_karsilastirma.png"
    fig.savefig(cikti_yolu, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Karşılaştırma grafiği kaydedildi: {cikti_yolu}")


def radar_chart_ciz(df: pd.DataFrame):
    """Modeller için radar chart (örümcek ağı) çiz."""
    if df is None or df.empty or len(df) > 5:
        return  # Çok model varsa karmaşık olur
    
    try:
        from math import pi
        
        metrikler = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Cohen Kappa']
        metrikler_mevcut = [m for m in metrikler if m in df.columns]
        
        if len(metrikler_mevcut) < 3:
            return
        
        # Radar chart için açılar
        angles = [n / float(len(metrikler_mevcut)) * 2 * pi for n in range(len(metrikler_mevcut))]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for idx, (_, row) in enumerate(df.iterrows()):
            if idx >= len(colors):
                break
            
            values = [row[m] for m in metrikler_mevcut]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=f"{row['Tip']} ({row['Tarih']})", color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrikler_mevcut)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.title('Model Performans Karşılaştırması (Radar Chart)', 
                 size=14, weight='bold', pad=20)
        
        cikti_yolu = GORSELLER_KLASORU / "model_radar_chart.png"
        fig.savefig(cikti_yolu, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Radar chart kaydedildi: {cikti_yolu}")
        
    except Exception as e:
        print(f"⚠️  Radar chart çizilemedi: {e}")


def main():
    """Ana fonksiyon."""
    print("\n" + "="*70)
    print("📊 MODEL KARŞILAŞTIRMA ANALİZİ")
    print("="*70)
    
    # Modelleri bul
    modeller = model_listesi_al()
    
    if not modeller:
        print("\n❌ Hiç eğitilmiş model bulunamadı!")
        print(f"   Aranan klasör: {MODELS_KLASORU}")
        print(f"\n💡 Önce model eğitin:")
        print(f"   python3 train.py --auto")
        return 1
    
    print(f"\n✓ {len(modeller)} model bulundu")
    
    # Karşılaştırma tablosu oluştur
    df = karsilastirma_tablosu_olustur(modeller)
    
    if df is None or df.empty:
        print("\n❌ Hiç metadata bulunamadı!")
        print("   Modeller metadata olmadan eğitilmiş olabilir.")
        return 1
    
    # Tabloyu göster
    print(f"\n" + "="*70)
    print("📋 MODEL PERFORMANS TABLOSU")
    print("="*70 + "\n")
    
    # DataFrame'i güzel formatla
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print(df.to_string(index=False))
    
    # En iyi modeller
    print(f"\n" + "="*70)
    print("🏆 EN İYİ MODELLER")
    print("="*70)
    
    metrikler = ['Accuracy', 'F1-Score', 'Cohen Kappa']
    for metrik in metrikler:
        if metrik in df.columns:
            en_iyi = en_iyi_model_sec(df, metrik)
            en_iyi_deger = df[df['Model'] == en_iyi][metrik].values[0]
            print(f"\n{metrik:15s}: {en_iyi} ({en_iyi_deger:.4f})")
    
    # Grafikler
    print(f"\n" + "="*70)
    print("📊 GRAFİKLER OLUŞTURULUYOR")
    print("="*70 + "\n")
    
    karsilastirma_grafigi_ciz(df)
    radar_chart_ciz(df)
    
    # CSV kaydet
    csv_yolu = GORSELLER_KLASORU / "model_karsilastirma.csv"
    df.to_csv(csv_yolu, index=False, encoding='utf-8')
    print(f"✓ Karşılaştırma tablosu kaydedildi: {csv_yolu}")
    
    print(f"\n" + "="*70)
    print("✅ ANALİZ TAMAMLANDI")
    print("="*70)
    print(f"\n📁 Çıktılar: {GORSELLER_KLASORU}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
