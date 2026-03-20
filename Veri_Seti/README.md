# Veri Seti Yapisi

Bu klasor, projenin bekledigi MRI veri yerlesimini tanimlar. Proje yalnizca `OriginalDataset` uzerinden calisacak sekilde sadelelestirilmistir.

## Beklenen Yapi

```text
Veri_Seti/
`-- OriginalDataset/
    |-- NonDemented/
    |-- VeryMildDemented/
    |-- MildDemented/
    `-- ModerateDemented/
```

## Sinif Adlari

Asagidaki klasor adlari proje boyunca sabit kabul edilir:

- `NonDemented`
- `VeryMildDemented`
- `MildDemented`
- `ModerateDemented`

## Kullanim Politikasi

- `OriginalDataset`: tek kaynak veri dizini
- `train`, `validation` ve `test`: ayni original veri kaynagindan uretilir
- Train tarafinda yalnizca transform tabanli augmentation kullanilir

## Geri Uyumluluk

Bazi araclar asagidaki yapilari da kabul eder:

```text
Veri_Seti/<SinifAdi>/
```

Ancak repo icindeki guncel beklenti, `Veri_Seti/OriginalDataset/<SinifAdi>/` yapisinin kullanilmasidir.
