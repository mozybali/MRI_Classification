#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline_quick_test.py
----------------------
Hızlı pipeline testi - dependency kontrolü ve temel işlevsellik doğrulaması
"""

import sys
from pathlib import Path

def test_imports():
    """Gerekli paketlerin yüklü olup olmadığını kontrol et."""
    print("\n" + "="*70)
    print("PAKET KONTROLÜ")
    print("="*70)
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'scipy': 'scipy',
        'skimage': 'scikit-image',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm',
    }
    
    optional_packages = {
        'SimpleITK': 'SimpleITK',
    }
    
    all_ok = True
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package:20s} - Yüklü")
        except ImportError:
            print(f"✗ {package:20s} - EKSİK (pip install {package})")
            all_ok = False
    
    print("\nOpsiyonel Paketler:")
    for module, package in optional_packages.items():
        try:
            __import__(module)
            print(f"✓ {package:20s} - Yüklü")
        except ImportError:
            print(f"⚠ {package:20s} - Yok (bazı özellikler kullanılamaz)")
    
    return all_ok


def test_veri_seti():
    """Veri setinin varlığını kontrol et."""
    print("\n" + "="*70)
    print("VERİ SETİ KONTROLÜ")
    print("="*70)
    
    veri_klasoru = Path("../Veri_Seti")
    if not veri_klasoru.exists():
        print(f"✗ Veri seti bulunamadı: {veri_klasoru.absolute()}")
        return False
    
    siniflar = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
    toplam = 0
    
    for sinif in siniflar:
        sinif_klasoru = veri_klasoru / sinif
        if sinif_klasoru.exists():
            dosyalar = list(sinif_klasoru.glob("*.jpg")) + list(sinif_klasoru.glob("*.png"))
            sayi = len(dosyalar)
            toplam += sayi
            print(f"✓ {sinif:20s}: {sayi:5d} görüntü")
        else:
            print(f"✗ {sinif:20s}: Klasör bulunamadı")
    
    print(f"\nToplam: {toplam} görüntü")
    return toplam > 0


def test_modul():
    """Modül import'unu test et."""
    print("\n" + "="*70)
    print("MODÜL KONTROLÜ")
    print("="*70)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import ayarlar
        print("✓ ayarlar.py yüklendi")
        
        from goruntu_isleyici import GorselIsleyici
        print("✓ goruntu_isleyici.py yüklendi")
        
        from ozellik_cikarici import OzellikCikarici
        print("✓ ozellik_cikarici.py yüklendi")
        
        # Temel nesne oluşturma
        isleyici = GorselIsleyici()
        print("✓ GorselIsleyici nesnesi oluşturuldu")
        
        cikarici = OzellikCikarici()
        print("✓ OzellikCikarici nesnesi oluşturuldu")
        
        return True
    except Exception as e:
        print(f"✗ Modül yükleme hatası: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ana test fonksiyonu."""
    print("\n" + "="*70)
    print("MRI GÖRÜNTÜ İŞLEME - HIZLI TEST")
    print("="*70)
    
    results = {
        "Paket Kontrolü": test_imports(),
        "Veri Seti Kontrolü": test_veri_seti(),
        "Modül Kontrolü": test_modul(),
    }
    
    print("\n" + "="*70)
    print("TEST SONUÇLARI")
    print("="*70)
    
    for test_adi, sonuc in results.items():
        durum = "✓ BAŞARILI" if sonuc else "✗ BAŞARISIZ"
        print(f"{test_adi:25s}: {durum}")
    
    if all(results.values()):
        print("\n✓ Tüm testler başarılı! Pipeline hazır.")
        print("\nBir sonraki adım:")
        print("  python3 ana_islem.py")
        return 0
    else:
        print("\n✗ Bazı testler başarısız. Lütfen eksikleri giderin.")
        print("\nEksik paketleri yüklemek için:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
