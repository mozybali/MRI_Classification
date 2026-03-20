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

## Not

Repo icindeki guncel beklenti `Veri_Seti/OriginalDataset/<SinifAdi>/` yapisidir ve proje akisi bu yol uzerine sabitlenmistir.
