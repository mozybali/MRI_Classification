# Veri Seti Yapisi

Bu klasor, projenin bekledigi MRI veri yerlesimini tanimlar. Kod tabani hem kok veri yapisini hem de `AugmentedAlzheimerDataset` ve `OriginalDataset` ayrimini destekler.

## Beklenen Yapi

```text
Veri_Seti/
|-- AugmentedAlzheimerDataset/
|   |-- NonDemented/
|   |-- VeryMildDemented/
|   |-- MildDemented/
|   `-- ModerateDemented/
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

## Varsayilan Kullanim Politikasi

- `AugmentedAlzheimerDataset`: train + validation
- `OriginalDataset`: test

Bu politika hem `model/` tarafinda hem de `eda_analiz/` ile `goruntu_isleme/` tarafinda varsayilan davranisla uyumludur.

## Geri Uyumluluk

Bazi araclar asagidaki yapilari da kabul eder:

```text
Veri_Seti/<SinifAdi>/
```

Ancak repo icindeki guncel beklenti, ayri `AugmentedAlzheimerDataset` ve `OriginalDataset` klasorlerinin bulunmasidir.
