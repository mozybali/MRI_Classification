"""
Özellik Çıkarıcı Modülü Testleri
Tests for ozellik_cikarici.py module.

Bu dosya OzellikCikarici sınıfının fonksiyonlarını test eder.
Görüntülerden özellik çıkarma, CSV oluşturma ve ölçeklendirme işlemlerini doğrular.
"""

import sys
import importlib
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "goruntu_isleme"))

import ozellik_cikarici as oc_mod
from ozellik_cikarici import OzellikCikarici, veri_boluntule, veri_setini_bol_ve_olceklendir


def _grouped_features_df(prefix: str, per_class: int) -> pd.DataFrame:
    rows = []
    classes = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
    for class_index, class_name in enumerate(classes):
        for source_idx in range(per_class):
            rows.append(
                {
                    "dosya_adi": f"{prefix}_{class_name}_{source_idx}.png",
                    "feature1": float(class_index * 10 + source_idx),
                    "feature2": float(class_index * 100 + source_idx),
                    "sinif": class_name,
                    "etiket": class_index,
                    "tam_yol": f"/tmp/{prefix}_{class_name}_{source_idx}.png",
                }
            )
    return pd.DataFrame(rows)


class TestOzellikCikarici:
    """OzellikCikarici sınıfı için test suite."""

    def test_init(self):
        """Nesnenin doğru şekilde oluşturulduğunu kontrol et."""
        cikarici = OzellikCikarici()
        assert cikarici is not None
        assert cikarici.n_jobs >= 1

    def test_kaynak_id_belirle_parantezli_kopyalari_gruplar(self):
        """Veri setindeki '26 (19).jpg' tipindeki kopyalar aynı kaynağa bağlanmalı."""
        assert OzellikCikarici.kaynak_id_belirle("26 (19).jpg") == "26"
        assert OzellikCikarici.kaynak_id_belirle("26_aug2.png") == "26"
        assert OzellikCikarici.kaynak_id_belirle("26 (19)_aug2.png") == "26"

    def test_csv_olustur(self, test_dataset_structure, temp_output_dir):
        """Veri setinden özellikler çıkarılıp CSV'ye kaydedilmeli."""
        cikarici = OzellikCikarici()

        cikti_csv = temp_output_dir / "test_ozellikler.csv"
        df = cikarici.csv_olustur(test_dataset_structure, cikti_csv=cikti_csv)

        # DataFrame boş olmamalı (test görüntüleri var)
        assert not df.empty
        assert len(df) == 12  # 4 sınıf * 3 görüntü
        assert 'sinif' in df.columns
        assert 'etiket' in df.columns

    def test_scaling_minmax(self, sample_features_df, temp_output_dir):
        """MinMax ölçeklendirme: değerler [0, 1] aralığında olmalı."""
        cikarici = OzellikCikarici()

        csv_path = temp_output_dir / "features.csv"
        sample_features_df.to_csv(csv_path, index=False)

        scaled_df = cikarici.scaling_uygula(
            metod='minmax',
            giris_csv=csv_path,
            cikti_csv=temp_output_dir / "scaled.csv"
        )

        assert not scaled_df.empty

        numeric_cols = cikarici._sayisal_sutunlari_bul(scaled_df)
        for col in numeric_cols:
            if scaled_df[col].std() > 0:
                assert scaled_df[col].min() >= -0.01
                assert scaled_df[col].max() <= 1.01

    def test_scaling_robust(self, sample_features_df, temp_output_dir):
        """Robust ölçeklendirme başarıyla uygulanmalı."""
        cikarici = OzellikCikarici()

        csv_path = temp_output_dir / "features.csv"
        sample_features_df.to_csv(csv_path, index=False)

        scaled_df = cikarici.scaling_uygula(
            metod='robust',
            giris_csv=csv_path,
            cikti_csv=temp_output_dir / "scaled_robust.csv"
        )

        assert not scaled_df.empty

    def test_scaling_standard(self, sample_features_df, temp_output_dir):
        """Standard (Z-score) ölçeklendirme: ortalama ~0 olmalı."""
        cikarici = OzellikCikarici()

        csv_path = temp_output_dir / "features.csv"
        sample_features_df.to_csv(csv_path, index=False)

        scaled_df = cikarici.scaling_uygula(
            metod='standard',
            giris_csv=csv_path,
            cikti_csv=temp_output_dir / "scaled_standard.csv"
        )

        assert not scaled_df.empty

        numeric_cols = cikarici._sayisal_sutunlari_bul(scaled_df)
        for col in numeric_cols:
            if scaled_df[col].std() > 0:
                assert abs(scaled_df[col].mean()) < 1.0

    def test_istatistik_raporu(self, sample_features_df, temp_output_dir, capsys):
        """İstatistik raporu doğru çalışmalı."""
        cikarici = OzellikCikarici()

        csv_path = temp_output_dir / "features.csv"
        sample_features_df.to_csv(csv_path, index=False)

        cikarici.istatistik_raporu(csv_dosyasi=csv_path)
        captured = capsys.readouterr()
        output = captured.out
        assert "RAPOR" in output.upper()
        assert "Toplam görüntü sayısı".lower() in output.lower() or "Toplam goruntu sayisi".lower() in output.lower()
        assert "Sınıf dağılımı".lower() in output.lower() or "Sinif dagilimi".lower() in output.lower()

    def test_csv_with_empty_directory(self, tmp_path):
        """Boş dizinden CSV oluşturulduğunda boş DataFrame dönmeli."""
        cikarici = OzellikCikarici()
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        df = cikarici.csv_olustur(empty_dir)

        assert df.empty

    def test_jpeg_dosyasi_feature_extractiona_dahil(self, tmp_path):
        """.jpeg uzantılı dosyalar özellik çıkarmaya dahil olmalı."""
        cikarici = OzellikCikarici()

        # Sınıf klasörü oluştur
        sinif = "NonDemented"
        sinif_klasoru = tmp_path / sinif
        sinif_klasoru.mkdir()

        # .jpeg dosyası oluştur
        img = Image.fromarray(
            np.random.randint(50, 200, (64, 64), dtype=np.uint8), mode='L'
        )
        img.save(sinif_klasoru / "test_image.jpeg")

        # .jpg dosyası oluştur (karşılaştırma için)
        img.save(sinif_klasoru / "test_image2.jpg")

        cikti_csv = tmp_path / "test_ozellikler.csv"
        df = cikarici.csv_olustur(tmp_path, cikti_csv=cikti_csv)

        # Her iki dosya da dahil olmalı
        assert len(df) == 2
        dosya_adlari = df['dosya_adi'].tolist()
        assert "test_image.jpeg" in dosya_adlari
        assert "test_image2.jpg" in dosya_adlari


class TestVeriBoluntule:
    """Veri bölme fonksiyonu testleri."""

    def test_veri_boluntule_basic(self, sample_features_df, temp_output_dir):
        """Veri seti üç parçaya bölünmeli ve toplam korunmalı."""
        csv_path = temp_output_dir / "features_scaled.csv"
        sample_features_df.to_csv(csv_path, index=False)

        train_df, val_df, test_df = veri_boluntule(
            csv_dosyasi=csv_path,
            cikti_klasoru=temp_output_dir
        )

        assert not train_df.empty
        assert not val_df.empty
        assert not test_df.empty

        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(sample_features_df)

    def test_veri_boluntule_proportions(self, temp_output_dir):
        """Bölme oranları yaklaşık olarak doğru olmalı."""
        data = {
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'sinif': ['Class' + str(i % 4) for i in range(100)],
            'etiket': [i % 4 for i in range(100)]
        }
        df = pd.DataFrame(data)

        csv_path = temp_output_dir / "large_features.csv"
        df.to_csv(csv_path, index=False)

        train_df, val_df, test_df = veri_boluntule(
            csv_dosyasi=csv_path,
            cikti_klasoru=temp_output_dir
        )

        total = len(df)
        assert 0.60 <= len(train_df) / total <= 0.80
        assert 0.08 <= len(val_df) / total <= 0.25
        assert 0.08 <= len(test_df) / total <= 0.25

    def test_veri_boluntule_stratification(self, temp_output_dir):
        """Her sınıf tüm setlerde temsil edilmeli."""
        data = {
            'dosya_adi': [f"img_{i}.png" for i in range(40)],
            'feature1': np.random.rand(40),
            'sinif': ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10,
            'etiket': [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
        }
        df = pd.DataFrame(data)

        csv_path = temp_output_dir / "imbalanced.csv"
        df.to_csv(csv_path, index=False)

        train_df, val_df, test_df = veri_boluntule(
            csv_dosyasi=csv_path,
            cikti_klasoru=temp_output_dir
        )

        beklenen_siniflar = {'A', 'B', 'C', 'D'}
        for df_split in [train_df, val_df, test_df]:
            unique_classes = set(df_split['sinif'].unique())
            assert unique_classes == beklenen_siniflar

    def test_veri_boluntule_yetersiz_ornek_hatasi(self, temp_output_dir):
        """Yetersiz örnekle ValueError fırlatılmalı."""
        data = {
            'feature1': [1.0, 2.0],
            'sinif': ['A', 'B'],
            'etiket': [0, 1]
        }
        df = pd.DataFrame(data)

        csv_path = temp_output_dir / "tiny.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="yeterli"):
            veri_boluntule(csv_dosyasi=csv_path, cikti_klasoru=temp_output_dir)

    def test_veri_boluntule_uc_grupta_anlamli_hata_verir(self, temp_output_dir):
        """3 grup olduğunda ikinci split çökmeden anlamlı ValueError üretmeli."""
        data = {
            'feature1': [1.0, 2.0, 3.0],
            'sinif': ['A', 'B', 'C'],
            'etiket': [0, 1, 2]
        }
        df = pd.DataFrame(data)
        csv_path = temp_output_dir / "three_groups.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="gereken minimum"):
            veri_boluntule(csv_dosyasi=csv_path, cikti_klasoru=temp_output_dir)

    def test_veri_setini_bol_ve_olceklendir_basarisiz_boluntuleme(self, temp_output_dir):
        """veri_boluntule başarısız olursa veri_setini_bol_ve_olceklendir None dönmeli."""
        # Yetersiz veri ile CSV oluştur
        data = {
            'feature1': [1.0],
            'sinif': ['A'],
            'etiket': [0]
        }
        df = pd.DataFrame(data)
        csv_path = temp_output_dir / "single.csv"
        df.to_csv(csv_path, index=False)

        sonuc = veri_setini_bol_ve_olceklendir(
            csv_dosyasi=csv_path,
            cikti_klasoru=temp_output_dir
        )
        assert sonuc is None

    def test_veri_boluntule_augmented_trainval_ve_original_test_stratejisini_destekler(self, temp_output_dir):
        trainval_csv = temp_output_dir / "augmented.csv"
        test_csv = temp_output_dir / "original.csv"
        _grouped_features_df(prefix="aug", per_class=4).to_csv(trainval_csv, index=False)
        _grouped_features_df(prefix="orig", per_class=2).to_csv(test_csv, index=False)

        train_df, val_df, test_df = veri_boluntule(
            csv_dosyasi=trainval_csv,
            cikti_klasoru=temp_output_dir,
            test_csv_dosyasi=test_csv,
        )

        assert set(train_df["dosya_adi"]).isdisjoint(set(test_df["dosya_adi"]))
        assert set(val_df["dosya_adi"]).isdisjoint(set(test_df["dosya_adi"]))
        assert sorted(train_df["etiket"].unique().tolist()) == [0, 1, 2, 3]
        assert sorted(val_df["etiket"].unique().tolist()) == [0, 1, 2, 3]
        assert sorted(test_df["etiket"].unique().tolist()) == [0, 1, 2, 3]

    def test_veri_boluntule_original_test_ile_kaynak_cakismasini_reddeder(self, temp_output_dir):
        trainval_csv = temp_output_dir / "augmented_overlap.csv"
        test_csv = temp_output_dir / "original_overlap.csv"
        _grouped_features_df(prefix="shared", per_class=4).to_csv(trainval_csv, index=False)
        _grouped_features_df(prefix="shared", per_class=2).to_csv(test_csv, index=False)

        with pytest.raises(ValueError, match="kaynak grup sizintisi"):
            veri_boluntule(
                csv_dosyasi=trainval_csv,
                cikti_klasoru=temp_output_dir,
                test_csv_dosyasi=test_csv,
            )


class TestEdgeCases:
    """Sınır durumları ve hata yönetimi testleri."""

    def test_scaling_with_constant_features(self, temp_output_dir):
        """Sabit özelliklerle ölçeklendirme hatasız çalışmalı."""
        cikarici = OzellikCikarici()

        data = {
            'constant_feature': [100] * 10,
            'variable_feature': np.random.rand(10),
            'sinif': ['A'] * 10,
            'etiket': [0] * 10
        }
        df = pd.DataFrame(data)

        csv_path = temp_output_dir / "constant.csv"
        df.to_csv(csv_path, index=False)

        scaled_df = cikarici.scaling_uygula(
            metod='minmax',
            giris_csv=csv_path,
            cikti_csv=temp_output_dir / "scaled_constant.csv"
        )

        assert not scaled_df.empty
        assert 'constant_feature' in scaled_df.columns

    def test_veri_boluntule_kaynak_grubu_korur(self, temp_output_dir):
        """Aynı kaynak görüntünün augmentasyonları farklı split'lere düşmemeli."""
        data = {
            'dosya_adi': ['img1.png', 'img1_aug1.png', 'img2.png', 'img2_aug1.png', 'img3.png', 'img3_aug1.png', 'img4.png', 'img4_aug1.png'],
            'feature1': np.random.rand(8),
            'sinif': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
            'etiket': [0, 0, 1, 1, 2, 2, 3, 3],
        }
        df = pd.DataFrame(data)
        csv_path = temp_output_dir / "grouped.csv"
        df.to_csv(csv_path, index=False)

        train_df, val_df, test_df = veri_boluntule(csv_dosyasi=csv_path, cikti_klasoru=temp_output_dir)
        splitler = [train_df, val_df, test_df]

        kaynak_splitleri = {}
        for idx, split_df in enumerate(splitler):
            for dosya_adi in split_df['dosya_adi']:
                kaynak = dosya_adi.replace('_aug1', '').replace('.png', '')
                kaynak_splitleri.setdefault(kaynak, set()).add(idx)

        assert all(len(split_setleri) == 1 for split_setleri in kaynak_splitleri.values())


class TestOzellikCikariciRegressions:
    """Regresyon testleri."""

    def test_csv_olustur_cikti_klasorunu_olusturur(self, test_dataset_structure, tmp_path, monkeypatch):
        """csv_olustur, hedef klasor yoksa once olusturabilmeli."""
        class DummyPool:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def imap(self, func, iterable):
                for item in iterable:
                    yield func(item)

        monkeypatch.setattr(oc_mod, "Pool", DummyPool)

        cikarici = OzellikCikarici()
        cikti_csv = tmp_path / "olmayan" / "alt" / "ozellikler.csv"

        df = cikarici.csv_olustur(test_dataset_structure, cikti_csv=cikti_csv)

        assert not df.empty
        assert cikti_csv.exists()

    def test_veri_boluntule_cikti_klasorunu_olusturur(self, temp_output_dir):
        """veri_boluntule, hedef klasor yoksa kayit oncesi olusturabilmeli."""
        data = {
            'dosya_adi': ['img1.png', 'img1_aug1.png', 'img2.png', 'img2_aug1.png', 'img3.png', 'img3_aug1.png', 'img4.png', 'img4_aug1.png'],
            'feature1': np.random.rand(8),
            'sinif': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
            'etiket': [0, 0, 1, 1, 2, 2, 3, 3],
        }
        csv_path = temp_output_dir / "grouped_for_output.csv"
        pd.DataFrame(data).to_csv(csv_path, index=False)

        cikti_klasoru = temp_output_dir / "olmayan" / "splitler"
        train_df, val_df, test_df = veri_boluntule(csv_dosyasi=csv_path, cikti_klasoru=cikti_klasoru)

        assert not train_df.empty
        assert not val_df.empty
        assert not test_df.empty
        assert (cikti_klasoru / "egitim.csv").exists()
        assert (cikti_klasoru / "dogrulama.csv").exists()
        assert (cikti_klasoru / "test.csv").exists()

    def test_modul_paket_olarak_import_edilebilir(self):
        """goruntu_isleme.ozellik_cikarici paket import'u ile yuklenebilmeli."""
        modul = importlib.import_module("goruntu_isleme.ozellik_cikarici")
        assert hasattr(modul, "OzellikCikarici")
