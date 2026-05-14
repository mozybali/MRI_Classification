# Beyin MRI Görüntülerinden Demans Sınıflandırması: ResNet18 ve XGBoost Karşılaştırmalı Analiz

**Proje:** MRI Classification  
**Modeller:** ResNet18 (PyTorch, Transfer Öğrenme) · XGBoost (El Yapımı Özellikler)  
**Tarih:** Mayıs 2026  
**Veri Seti:** Kaggle Augmented Alzheimer MRI Dataset (33.984 görüntü)

---

## 1. Giriş

Alzheimer hastalığı, demansın en yaygın alt tipi olup dünya genelinde 57 milyondan fazla insanı etkilemektedir; 2050 yılına kadar bu sayının 139 milyona ulaşması beklenmektedir. Hastalığın yavaş ilerleyen doğası ve erken dönemde belirti vermemesi, klinik teşhisi güçleştirmektedir. Bu nedenle nesnel ve tekrarlanabilir tanı araçlarına olan ihtiyaç giderek artmaktadır.

Manyetik rezonans görüntüleme (MRI), Alzheimer'a bağlı beyin yapısal değişikliklerini — hippokampal atrofi, kortikal incelme, ventrikül genişlemesi — görselleştirmek için tercih edilen yöntemdir. Ancak bu değişikliklerin radyolojik yorumu uzmanlık gerektirmekte ve yorumlayıcılar arasında tutarsızlıklara yol açabilmektedir.

**Bu çalışmanın amacı**, 2D beyin MRI görüntülerinden demans şiddetini dört sınıfa (NonDemented, VeryMildDemented, MildDemented, ModerateDemented) otomatik olarak sınıflandırmak ve bu amaçla iki farklı makine öğrenmesi yaklaşımını karşılaştırmaktır:

1. **ResNet18** — ImageNet ağırlıklarıyla başlatılmış derin öğrenme modeli (transfer öğrenme)
2. **XGBoost** — HOG/LBP/GLCM tabanlı el yapımı özellikler kullanan gradyan artırma modeli

Her iki model de Optuna Bayesian TPE ile hiperparametre optimizasyonuna tabi tutulmuş; böylece adil ve sistematik bir karşılaştırma ortamı sağlanmıştır.

---

## 2. Literatür

### Derin Öğrenme Tabanlı Çalışmalar

| Çalışma | Model | Veri | Sonuç |
|---|---|---|---|
| Kaur ve ark. (2025) | ResNet-50 + EfficientNet-B3 (Topluluk) | 33.984 görüntü (augmente) | %99,32 doğruluk |
| Kırtay & Koçak (2024) | ResNet50 + SVM | 6.400 görüntü (orijinal) | %98 doğruluk |
| PMC (2025) | InceptionResNetV2 | 6.735 görüntü | %99 doğruluk |

CNN tabanlı derin öğrenme mimarileri, tıbbi görüntülemede tutarlı biçimde yüksek performans sergilemektedir. ResNet ailesinin artık (residual) bağlantı yapısı, derin ağlarda gradyan kaybı sorununu çözerek güçlü özellik temsili öğrenilmesini sağlar. ResNet18, 11,7 milyon parametreyle ResNet50'ye kıyasla çok daha hızlı eğitim ve çıkarım süreleri sunarken rekabetçi doğruluk değerlerine ulaşmaktadır.

Transfer öğrenme, tıbbi görüntülemedeki sınırlı veri problemini aşmak için yaygın bir strateji olup ImageNet ağırlıklarıyla başlatılan modeller MRI gibi farklı alanlarda da etkili biçimde ince ayar yapılabilmektedir.

### Geleneksel Makine Öğrenmesi Tabanlı Çalışmalar

| Çalışma | Model | Veri | Sonuç |
|---|---|---|---|
| Openbio. (2022) | XGBoost (morfoloji özellikleri) | ADNI veri tabanı | %92,31 doğruluk, AUC = 0,9543 |

XGBoost gibi gradyan artırma yöntemleri, özellikle sınırlı eğitim verisi ve yorumlanabilirlik gerektiren klinik senaryolarda değerli bir alternatif sunmaktadır. HOG (Histogram of Oriented Gradients), LBP (Local Binary Patterns) ve GLCM (Gray Level Co-occurrence Matrix) gibi el yapımı özellik çıkarım yöntemleri, doku ve şekil bilgisini sayısal vektörler olarak temsil etmektedir.

### Bu Çalışmanın Katkısı

Literatürdeki çalışmaların büyük çoğunluğu ya yalnızca derin öğrenme ya da yalnızca geleneksel makine öğrenmesi yaklaşımını ele almaktadır. Bu çalışma, **aynı veri seti, aynı ön işleme pipeline'ı ve eşdeğer HPO stratejisiyle** her iki paradigmayı doğrudan karşılaştırarak literatüre katkı sunmaktadır.

---

## 3. Model

### 3.1 Veri Seti

Kaggle platformunda kamuya açık olarak sunulan **Augmented Alzheimer MRI Dataset** kullanılmıştır. Orijinal 6.400 görüntü; döndürme, yakınlaştırma, yatay çevirme ve parlaklık değişimi gibi veri artırma teknikleriyle 33.984 görüntüye genişletilmiştir.

| Sınıf | Orijinal | Augmente Sonrası | Oran |
|---|---:|---:|---:|
| NonDemented | 3.200 | 9.600 | %28,2 |
| VeryMildDemented | 2.240 | 8.960 | %26,4 |
| MildDemented | 896 | 8.960 | %26,4 |
| ModerateDemented | 64 | 6.464 | %19,0 |
| **Toplam** | **6.400** | **33.984** | **%100** |

Augmente türevler yalnızca eğitim/doğrulama tarafında tutulmuş; **test seti (%15) yalnızca orijinal görüntülerden** oluşturularak gerçek genelleme kapasitesinin ölçülmesi güvence altına alınmıştır.

**Görüntü Ön İşleme (her iki model için aynı pipeline):**
1. Gri ton dönüşümü (OpenCV)
2. Kenar artefakt tespiti ve temizliği (bağlantılı bileşen analizi)
3. Percentile clipping (0,5–99,5) ve 0–255 normalleştirme
4. CLAHE ile kontrast iyileştirme (`clip_limit = 2.0`)
5. Hedef boyuta en-boy oranı korunarak yeniden boyutlandırma

### 3.2 Algoritmalar

#### ResNet18 (Derin Öğrenme)

`torchvision.models.resnet18` omurgası ImageNet ağırlıklarıyla başlatılmış, son fully-connected katmanı 4 sınıflı çıktı verecek biçimde değiştirilmiştir. Classifier head'e dropout uygulanmış; optimizer olarak AdamW, scheduler olarak ReduceLROnPlateau kullanılmıştır. Eğitim sırasında online augmentasyon uygulanmış; anatomik lateralite kaygısıyla yatay çevirme devre dışı bırakılmıştır.

#### XGBoost (El Yapımı Özellik Tabanlı)

Her görüntüden toplam **4.451 boyutlu** özellik vektörü çıkarılmıştır:
- **HOG** — Yönelim gradyanı histogramları (beyin morfolojisindeki doku değişimlerini temsil eder)
- **LBP** — Lokal ikili örüntüler (doku mikro-yapısı)
- **GLCM** — Gri seviye eş-oluşum matrisi (uzamsal ilişkiler)
- **İstatistiksel özellikler** — Ortalama, standart sapma, yüzdelik dilimler

Sınıf dengesizliğine karşı `class_balance = balanced` parametresiyle ağırlıklandırma uygulanmıştır.

---

## 4. Deneysel Çalışmalar

### 4.1 Hiperparametre Optimizasyonu

Her iki model de **Optuna Bayesian TPE** yöntemiyle optimize edilmiştir.

| Özellik | ResNet18 | XGBoost |
|---|---|---|
| Toplam trial | 100 (95 tamamlandı) | 150 (150 tamamlandı) |
| CV katmanı | 3-fold | 3-fold |
| Hedef metrik | Makro F1 | Makro F1 |
| En iyi trial | #90 | #138 |
| En iyi val F1 (CV) | **0,9797 ± 0,0034** | 0,8787 ± 0,0085 |

**ResNet18 — En İyi Hiperparametreler:**

| Parametre | Değer |
|---|---|
| `batch_size` | 128 |
| `image_size` | 192 |
| `lr` | 0,000301 |
| `dropout` | 0,1805 |
| `weight_decay` | 0,000861 |
| `label_smoothing` | 0,00235 |
| `best_epoch` | 23 |

**XGBoost — En İyi Hiperparametreler:**

| Parametre | Değer |
|---|---|
| `n_estimators` | 428 |
| `max_depth` | 8 |
| `learning_rate` | 0,0966 |
| `subsample` | 0,4332 |
| `colsample_bytree` | 0,6743 |
| `gamma` | 0,0037 |
| `min_child_weight` | 4 |
| `max_delta_step` | 14 |
| `image_size` | 224 |
| `class_balance` | balanced |
| `best_iteration` | 427 |

### 4.2 ResNet18 — Test Sonuçları

![ResNet18 Normalize Karmaşıklık Matrisi](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/6d2aa772-e8d6-496f-9e42-5da38da56fe0)

*Şekil 1. ResNet18 Normalize Karmaşıklık Matrisi — NonDemented %98, VeryMildDemented %100, MildDemented %100, ModerateDemented %100 sınıf içi doğruluk.*

| Sınıf | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NonDemented | 1,000 | 0,981 | 0,990 | 622 |
| VeryMildDemented | 0,968 | 1,000 | 0,984 | 270 |
| MildDemented | 0,982 | 1,000 | 0,991 | 163 |
| ModerateDemented | 1,000 | 1,000 | 1,000 | 10 |
| **Makro Ortalama** | **0,987** | **0,995** | **0,991** | **1.065** |

![ResNet18 ROC ve Precision-Recall Eğrileri](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/6bde416c-30ff-4743-9d98-8b599bfceb4f)

*Şekil 2. ResNet18 One-vs-Rest ROC ve Precision-Recall Eğrileri — Tüm sınıflarda AUC ≥ 0,999 ve AP ≥ 0,998.*

| Sınıf | ROC-AUC | AP |
|---|---:|---:|
| NonDemented | 1,000 | 1,000 |
| VeryMildDemented | 0,999 | 0,998 |
| MildDemented | 1,000 | 1,000 |
| ModerateDemented | 1,000 | 1,000 |
| **Makro AUC-OVR** | **0,9998** | **0,9995** |

### 4.3 XGBoost — Test Sonuçları

![XGBoost Özellik Önemi](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/fa4dd823-4221-4932-b53e-4072d2feb40d)

*Şekil 3. XGBoost Özellik Önemi — HOG grubu toplam önem skorunun %80'inden fazlasını oluştururken GLCM ve LBP tamamlayıcı katkı sunmaktadır.*

![XGBoost Normalize Karmaşıklık Matrisi](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/73a49750-177f-4b61-b049-4e6aa646d06c)

*Şekil 4. XGBoost Normalize Karmaşıklık Matrisi — NonDemented %94, VeryMildDemented %94, MildDemented %91, ModerateDemented %100 sınıf içi recall.*

| Sınıf | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NonDemented | 0,965 | 0,941 | 0,953 | 622 |
| VeryMildDemented | 0,870 | 0,941 | 0,904 | 270 |
| MildDemented | 0,955 | 0,914 | 0,934 | 163 |
| ModerateDemented | 0,909 | 1,000 | 0,952 | 10 |
| **Makro Ortalama** | **0,925** | **0,949** | **0,936** | **1.065** |

![XGBoost ROC ve Precision-Recall Eğrileri](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/0ecfb997-b882-4bb8-aace-afaf15a232cb)

*Şekil 5. XGBoost One-vs-Rest ROC ve Precision-Recall Eğrileri — Makro AUC-OVR = 0,992, Makro AP = 0,985.*

### 4.4 Model Karşılaştırması

| Metrik | ResNet18 | XGBoost |
|---|---:|---:|
| Test Doğruluğu | **%98,87** | %93,71 |
| Makro F1 (Test) | **0,991** | 0,936 |
| Makro F1 (CV, HPO) | **0,9797 ± 0,0034** | 0,8787 ± 0,0085 |
| Makro AUC-OVR | **0,9998** | 0,9920 |
| Makro AP | **0,9995** | 0,9850 |
| NonDemented F1 | **0,990** | 0,953 |
| VeryMildDemented F1 | **0,984** | 0,904 |
| MildDemented F1 | **0,991** | 0,934 |
| ModerateDemented F1 | **1,000** | 0,952 |
| Ortalama tahmin güveni | **0,955** | 0,863 |
| Yüksek güvenli hata sayısı | **3** | 2 |

**Literatürle Karşılaştırma:**

| Çalışma | Model | Doğruluk / F1 |
|---|---|---|
| Kaur ve ark. (2025) | ResNet-50 + EfficientNet-B3 (Topluluk) | %99,32 |
| Kırtay & Koçak (2024) | ResNet50 + SVM | %98 |
| **Bu çalışma — ResNet18** | **ResNet18 (Transfer Learning)** | **%98,87 / F1=0,991** |
| PMC (2025) | InceptionResNetV2 | %99 |
| **Bu çalışma — XGBoost** | **XGBoost (HOG+LBP+GLCM, class-balanced)** | **F1=0,936** |
| Openbio. (2022) | XGBoost (morfoloji özellikleri, ADNI) | %92,31 |

---

## 5. Sonuç

### Neden Böyle Bir Uygulamaya İhtiyaç Var?

Alzheimer hastalığının erken evreleri (NonDemented → VeryMildDemented) görsel olarak birbirine çok yakındır ve uzman radyologlar arasında bile yorumlama farklılıkları gözlemlenebilmektedir. Otomatik MRI sınıflandırma sistemleri; klinik karar destek aracı olarak tanı süreçlerini hızlandırabilir, standartlaştırabilir ve özellikle uzman erişiminin kısıtlı olduğu bölgelerde kritik bir rol üstlenebilir.

### Bu Çalışmadan Elde Edilen Kazanımlar

- **ResNet18**, hafif bir derin öğrenme mimarisi olmasına karşın yalnızca 23 epoch eğitimle test setinde **%98,87 doğruluk** ve **makro F1 = 0,991** elde etmiştir. Bu sonuç, toplu ve karmaşık modellere (ResNet50 + EfficientNet topluluğu) yakın bir performansı tek bir modelle elde etmenin mümkün olduğunu göstermektedir.
- **XGBoost**, el yapımı HOG/LBP/GLCM özellikleri ve sınıf ağırlıklandırmasıyla test setinde **%93,71 doğruluk** ve **makro F1 = 0,936** elde etmiştir. Geleneksel XGBoost literatürüne (%92,31) kıyasla daha iyi bir sonuç olup yorumlanabilirlik ve hesaplama verimliliği avantajını korumaktadır.
- **ModerateDemented** (ileri evre) sınıfında her iki model de recall = 1,000 elde etmiştir; bu sınıfın klinik önemi göz önüne alındığında kritik bir bulgudur (yanlış negatif = 0).

### En İyi Sonucu Veren Algoritma

**ResNet18**, tüm metriklerde XGBoost'u geride bırakmıştır. Bunun temel nedeni, CNN'lerin hiyerarşik özellik öğrenimi kapasitesidir: alt katmanlar kenar ve doku gibi düşük düzeyli özellikleri, üst katmanlar ise Alzheimer'a özgü anatomik örüntüleri (hippokampal atrofi, gri madde kaybı) öğrenmektedir. El yapımı HOG/LBP/GLCM özellikleri bu hiyerarşik temsilleri yakalamakta yetersiz kalmaktadır.

Bununla birlikte **XGBoost**, klinisyenler için yorumlanabilirlik gerektiren ya da GPU altyapısı bulunmayan senaryolarda değerli bir tamamlayıcı model olma niteliğini korumaktadır.

### Gelecekte Neler Yapılabilir?

- **ResNet18 + XGBoost topluluk modeli:** İki yaklaşımın güçlü yönlerini birleştiren hibrit bir sistem geliştirilebilir.
- **3D volumetrik MRI:** 2D dilimler yerine 3D MRI verisinin kullanımı anatomik bağlamı daha kapsamlı biçimde temsil edebilir.
- **Çok modlu veri:** Görüntü verisi ile yaş, cinsiyet ve bilişsel test sonuçları gibi klinik verilerin bütünleştirilmesi tahmin doğruluğunu artırabilir.
- **Harici doğrulama:** Bağımsız klinik kohortlarla prospektif değerlendirme, modelin gerçek dünya genellenebilirliğini doğrulamak için gereklidir.

---

## Kaynaklar

1. Kaur ve ark. (2025) — [Intelligent Alzheimer's diagnosis](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1619228/full), Frontiers in Medicine
2. Alzheimer's Disease International — [World Alzheimer Report 2024](https://www.alzint.org/u/World-Alzheimer-Report-2024.pdf)
3. WHO — [Dementia Fact Sheet](https://www.who.int/health-topics/dementia)
4. PMC (2025) — [Classifying Alzheimer's with deep CNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC12216300/)
5. Openbiotechnology (2022) — [XGBoost for MRI Alzheimer's](https://openbiotechnologyjournal.com/VOLUME/16/ELOCATOR/e187407072208300/FULLTEXT/)
6. Kırtay & Koçak (2024) — [Transfer Learning in Dementia Classification](https://dergipark.org.tr/en/download/article-file/3386757)
7. Optuna — [Hyperparameter Optimization Framework](https://optuna.readthedocs.io)
