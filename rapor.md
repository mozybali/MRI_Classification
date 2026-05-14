# Beyin MRI Görüntülerinden Demans Sınıflandırması: ResNet18 ve XGBoost Karşılaştırmalı Analiz
**Proje:** MRI Classification  
**Modeller:** ResNet18 (PyTorch, Transfer Öğrenme) · XGBoost (El Yapımı Özellikler)  
**Tarih:** Mayıs 2026  
**Veri Seti:** Kaggle Augmented Alzheimer MRI Dataset (33.984 görüntü)

***
## Özet
Bu çalışmada, augmente edilmiş 2D beyin MRI görüntülerinden demans şiddetini dört sınıfa (NonDemented, VeryMildDemented, MildDemented, ModerateDemented) sınıflandırmak amacıyla iki farklı yaklaşım karşılaştırılmıştır: (1) ImageNet ağırlıklarıyla başlatılmış **ResNet18** derin öğrenme modeli ve (2) HOG/LBP/GLCM tabanlı el yapımı özellikler kullanan **XGBoost** gradyan artırma modeli. Her iki model de Optuna Bayesian TPE ile hiperparametre optimizasyonuna tabi tutulmuştur.

**ResNet18** (Tuned): Test seti doğruluğu **%98,87**, makro F1 = **0,9912**, makro AUC-OVR = **0,9998**  
**XGBoost** (Tuned): Test seti doğruluğu **%93,71**, makro F1 = **0,9358**, makro AUC-OVR = **0,9920**, 3-fold CV HPO F1 = **0,8787**

Derin öğrenme yaklaşımı, uçtan uca özellik öğrenimi sayesinde el yapımı özellik tabanlı modeli tüm metriklerde geride bırakmıştır. Bununla birlikte XGBoost, yorumlanabilirlik ve hesaplama verimliliği avantajlarıyla tamamlayıcı bir referans model olma niteliği taşımaktadır.

***
## 1. Giriş
Alzheimer hastalığı (AH), demansın en yaygın alt tipi olup dünya genelinde 57 milyondan fazla insanı etkilemektedir; bu sayının 2050 yılına kadar 139 milyona ulaşması beklenmektedir. Hastalığın erken ve doğru teşhisi, klinik müdahale planlaması ve hasta bakımı açısından kritik önem taşımaktadır. Manyetik rezonans görüntüleme (MRI), Alzheimer'a bağlı beyin yapısal değişikliklerini — hippokampal atrofi, kortikal incelme, ventrikül genişlemesi — görselleştirmek için tercih edilen yöntem olma özelliğini korumaktadır.[^1][^2][^3]

Son yıllarda CNN tabanlı derin öğrenme mimarileri tıbbi görüntülemede hâkim konuma gelmiş; özellik mühendisliğine dayalı geleneksel yöntemlere kıyasla tutarlı biçimde üstün sonuçlar bildirmiştir. Bununla birlikte XGBoost gibi gradyan artırma yöntemleri, özellikle sınırlı veri koşullarında rekabetçi sonuçlar ve yorumlanabilirlik avantajı sunmaya devam etmektedir. Bu çalışma, her iki paradigmayı aynı veri seti, aynı ön işleme pipeline'ı ve eşdeğer hiperparametre optimizasyon stratejisiyle değerlendirerek doğrudan bir karşılaştırma sunmaktadır.[^4][^5]

***
## 2. İlgili Çalışmalar
Kaur ve ark. (2025), Frontiers in Medicine'de yayımlanan çalışmalarında aynı augmente veri setini (33.984 görüntü) kullanarak ResNet-50 ve EfficientNet-B3'ten oluşan bir topluluk modeli geliştirmiş ve %99,32 genel doğruluk bildirmiştir. Kırtay ve Koçak (2024), orijinal 6.400 görüntülük veri seti üzerinde ResNet50 transfer öğrenme + SVM kombinasyonunda %98 doğruluğa ulaşmıştır. Geleneksel makine öğrenmesi alanında, ADNI veri tabanından çıkarılan beyin morfoloji özelliklerini kullanan bir XGBoost modeli %92,31 doğruluk ve 0,9543 AUC değerleri elde etmiştir.[^5][^1][^6]

ResNet ailesi, artık (residual) bağlantılar sayesinde derin ağlarda karşılaşılan gradyan kaybı sorununu aşarak ImageNet gibi büyük veri setlerinde güçlü özellik temsilleri öğrenmektedir. ResNet18'in 11,7 milyon parametreyle eğitim süresi ve çıkarım hızı açısından ResNet50'ye kıyasla belirgin avantaj sunduğu gösterilmiştir. Transfer öğrenme ise tıbbi görüntülemedeki sınırlı veri sorununu aşmak için standart bir strateji haline gelmiştir.[^7][^8][^9]

***
## 3. Materyal ve Yöntem
### 3.1 Veri Seti
Bu çalışmada Kaggle platformunda kamuya açık olarak sunulan Alzheimer MRI veri setinin **augmente edilmiş versiyonu** kullanılmıştır. Orijinal 6.400 görüntü; döndürme, yakınlaştırma, yatay çevirme ve parlaklık değişimi gibi veri artırma teknikleriyle genişletilerek **33.984 görüntülük** bir havuz oluşturulmuştur.[^1][^10]

| Sınıf | Orijinal Görüntü | Augmente Sonrası | Oran (Augmente) |
|---|---:|---:|---:|
| NonDemented | 3.200 | 9.600 | %28,2 |
| VeryMildDemented | 2.240 | 8.960 | %26,4 |
| MildDemented | 896 | 8.960 | %26,4 |
| ModerateDemented | 64 | 6.464 | %19,0 |
| **Toplam (Augmente)** | **6.400** | **33.984** | **%100** |

Orijinal veri setinde yalnızca 64 görüntüyle temsil edilen ModerateDemented sınıfı, augmentasyon sonrasında 6.464 görüntüye ulaşarak sınıf dengesizliği önemli ölçüde azaltılmıştır. Augmente türevler yalnızca trainval tarafında tutulmuş; test seti (%15) yalnızca orijinal görüntülerden oluşturulmuştur. Bu yaklaşım, test değerlendirmesinin gerçek genelleme kapasitesini yansıtmasını güvence altına almaktadır.[^11][^10]
### 3.2 Görüntü Ön İşleme
Ham 2D MRI görüntüleri her iki model için aynı pipeline ile işlenmiştir:

1. **Gri ton dönüşümü** — OpenCV ile tek kanallı yoğunluk görüntüsü.
2. **Kenar artefakt tespiti ve temizliği** — CLAHE öncesinde bağlantılı bileşen analizi ile parlak kenar artefaktları giderilmiştir.
3. **Percentile clipping** — Foreground maskesi içinde (0,5–99,5) aralığında yoğunluk kırpması ve 0–255 normalleştirme.
4. **CLAHE** — `clip_limit = 2.0` ile uyarlamalı kontrast iyileştirme; beyin dokusundaki ince yapısal ayrıntıları görünür kılmaktadır.[^12][^13]
5. **Resize/Padding** — `192×192` hedef boyutuna en-boy oranı korunarak yeniden boyutlandırma.
### 3.3 Model Mimarileri
#### ResNet18 (Derin Öğrenme)
`torchvision.models.resnet18` omurgası, ImageNet ağırlıklarıyla başlatılmış (`pretrained=True`) ve son fully-connected katmanı 4 sınıflı çıktı verecek biçimde değiştirilmiştir. Classifier head'e dropout (`p=0.18`) uygulanmıştır. Kayıp fonksiyonu olarak Cross Entropy (label smoothing = 0.0023), optimizer olarak AdamW, scheduler olarak ReduceLROnPlateau kullanılmıştır. Eğitim sırasında online augmentasyon (döndürme 5°, color jitter 0.057) uygulanmıştır; yatay çevirme (`hflip_p=0.0`) anatomik lateralite kaygısıyla devre dışı bırakılmıştır.[^8]

#### XGBoost (El Yapımı Özellik Tabanlı)
Her görüntüden toplam **4.451 boyutlu** özellik vektörü çıkarılmıştır: HOG (Histogram of Oriented Gradients), LBP (Local Binary Patterns), GLCM (Gray Level Co-occurrence Matrix) ve istatistiksel özellikler (ortalama, standart sapma, yüzdelik dilimler). Özellik önemi analizinde HOG grubunun toplam önem skorunun %80'inden fazlasını oluşturduğu görülmüştür.[^14][^15]
### 3.4 Hiperparametre Optimizasyonu
Her iki model de Optuna Bayesian TPE yöntemiyle optimize edilmiştir. Arama stratejileri aşağıda karşılaştırılmaktadır:[^16]

| Özellik | ResNet18 HPO | XGBoost HPO |
|---|---|---|
| Toplam trial | 100 (95 tamamlandı) | 150 (150 tamamlandı) |
| CV katman sayısı | 3-fold | 3-fold |
| Hedef metrik | Makro F1 | Makro F1 |
| Arama uzayı | lr, dropout, batch_size, image_size, weight_decay, scheduler_factor/patience, loss, label_smoothing, color_jitter, rotation, hflip_p, pretrained | n_estimators, max_depth, lr, subsample, colsample_bytree, reg_lambda, reg_alpha, gamma, min_child_weight, max_delta_step, image_size |
| En iyi trial | #90 | #138 |
| En iyi val F1 (CV) | **0,9797 ± 0,0034** | 0,8787 ± 0,0085 |

***
## 4. ResNet18 — HPO Süreci
### 4.1 Parametre Önem Analizi
![ResNet HPO Parametre Önemleri](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/5c01d110-d848-4b8a-9095-ad95a4c8ea31)

*Şekil 1. ResNet18 HPO Parametre Önem Analizi — `lr` (0,33) ve `dropout` (0,30) en belirleyici parametreler olarak öne çıkmıştır. Bunu `scheduler_patience` (0,09), `scheduler_factor` (0,09) ve `color_jitter` (0,07) takip etmektedir. Regularizasyon parametreleri (`rotation_degrees`, `batch_size`, `hflip_p`) nispeten düşük önem sergilemiştir.*
### 4.2 Optimizasyon Geçmişi
![ResNet HPO Optimizasyon Geçmişi](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/7bf270fd-8e86-428d-a8a6-56f0761ec41f)

*Şekil 2. ResNet18 HPO Optimizasyon Geçmişi — İlk triallarda ~0,83 olan nesne değeri (makro F1) yaklaşık 15. trial'dan itibaren hızlı bir yakınsama göstererek 0,97'nin üzerine çıkmıştır. En iyi değer (0,9797) 90. trial'da elde edilmiştir.*
### 4.3 Slice Plot
![ResNet HPO Slice Plot](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/d7573727-c9fe-4c6c-81fa-1b55ffa1868e)

*Şekil 3. ResNet18 Slice Plot — Her hiperparametrenin arama uzayı boyunca nesne değerine etkisi. Öğrenme hızı ve dropout'un optimal değer bantları çevresinde yüksek F1 değerlerinin yoğunlaştığı görülmektedir.*
### 4.4 Paralel Koordinat Grafiği
![ResNet HPO Paralel Koordinat](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/53b026c1-2a9d-44e8-9454-2bd063aa9274)

*Şekil 4. ResNet18 Paralel Koordinat Grafiği — Tüm hiperparametreler ve nesne değeri arasındaki çok boyutlu ilişki.*
### 4.5 En İyi Hiperparametre Konfigürasyonu
| Parametre | Değer |
|---|---|
| `batch_size` | 128 |
| `image_size` | 192 |
| `lr` | 0,000301 |
| `weight_decay` | 0,000861 |
| `scheduler_factor` | 0,5526 |
| `scheduler_patience` | 2 |
| `loss` | Cross Entropy |
| `label_smoothing` | 0,00235 |
| `dropout` | 0,1805 |
| `hflip_p` | 0,0 |
| `rotation_degrees` | 5° |
| `color_jitter` | 0,0568 |
| `pretrained` | True |
| **val_f1_mean (3-fold)** | **0,9797** |
| **val_f1_std** | **±0,0034** |
| **best_epoch** | 23 |

***
## 5. ResNet18 — Eğitim Süreci
### 5.1 Eğitim Eğrileri
![ResNet Eğitim Eğrileri](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/74208acc-9565-4c2a-8cee-8d4470a6d7f1)

*Şekil 5. ResNet18 Eğitim Eğrileri — Sol üst: Loss eğrisi (1,02'den 0,089'a düzenli iniş). Sağ üst: Accuracy eğrisi (1. epoch %50'den 23. epoch %100'e). Sol alt: Makro F1 eğrisi. Sağ alt: Precision/Recall eğrileri. 23 epoch boyunca tutarlı yakınsama gözlemlenmiştir; 17–18. epoch civarında kısa bir dalgalanma sonrası model son düzlüğe ulaşmıştır.*

Eğitim train loss değeri 1. epoch'ta 1,025'ten başlayarak 23. epoch'ta 0,089'a inmiş; train accuracy ise %50,2'den %100'e yükselmiştir. Bu yakınsama paterni ImageNet ağırlıklarıyla başlatılan transfer öğrenmenin hızlı uyum sağlama kapasitesini yansıtmaktadır.[^7]

***
## 6. ResNet18 — Test Sonuçları
### 6.1 Karmaşıklık Matrisi
![ResNet Confusion Matrix](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/a898ea6c-3552-44d0-9e89-a2e0666335a0)

*Şekil 6. ResNet18 Karmaşıklık Matrisi — NonDemented: 622 örnekten 610 doğru, 12 hata. VeryMildDemented: 270 örnekten 270 doğru, 0 hata. MildDemented: 163 örnekten 163 doğru, 0 hata. ModerateDemented: 10 örnekten 10 doğru, 0 hata.*

![ResNet Normalized Confusion Matrix](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/6d2aa772-e8d6-496f-9e42-5da38da56fe0)

*Şekil 7. ResNet18 Normalize Karmaşıklık Matrisi — NonDemented %98, VeryMildDemented %100, MildDemented %100, ModerateDemented %100 sınıf içi doğruluk.*

| Sınıf | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NonDemented | 1,000 | 0,981 | 0,990 | 622 |
| VeryMildDemented | 0,968 | 1,000 | 0,984 | 270 |
| MildDemented | 0,982 | 1,000 | 0,991 | 163 |
| ModerateDemented | 1,000 | 1,000 | 1,000 | 10 |
| **Makro Ortalama** | **0,987** | **0,995** | **0,991** | **1.065** |
### 6.2 ROC ve Precision-Recall Eğrileri
![ResNet ROC PR Curves](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/6bde416c-30ff-4743-9d98-8b599bfceb4f)

*Şekil 8. ResNet18 One-vs-Rest ROC ve Precision-Recall Eğrileri — Tüm sınıflarda AUC ≥ 0,999 ve AP ≥ 0,998 değerleri elde edilmiştir.*

| Sınıf | ROC-AUC | AP (Precision-Recall) |
|---|---:|---:|
| NonDemented | 1,000 | 1,000 |
| VeryMildDemented | 0,999 | 0,998 |
| MildDemented | 1,000 | 1,000 |
| ModerateDemented | 1,000 | 1,000 |
| **Makro AUC-OVR** | **0,9998** | **0,9995** |
### 6.3 Sınıf Bazlı Performans Özeti
![ResNet Classification Summary](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/bfdc01bc-9f7f-4da5-bda0-823fa2e2cc33)

*Şekil 9. ResNet18 Sınıf Bazlı Performans — Tüm sınıflarda Precision, Recall ve F1 ~0,97 veya üzerinde.*
### 6.4 Tahmin Güveni Analizi
![ResNet Prediction Confidence](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/f7182073-40d7-4096-8b47-b565ce790403)

*Şekil 10. ResNet18 Tahmin Güveni Dağılımı — Doğru tahminler (yeşil) 0,90+ güven bandında yoğunlaşmıştır; yanlış tahminler orta güven bandında (~0,67 medyan) konumlanmaktadır. Yüksek güvenli hata sayısı yalnızca 3'tür.*

Ortalama tahmin güveni 0,9548 olup doğru tahminlerde 0,958, yanlış tahminlerde 0,670 olarak ölçülmüştür. Bu dağılım, modelin belirsizliğini güven skoru üzerinden doğru biçimde yansıttığını ortaya koymaktadır.

***
## 7. XGBoost — HPO Süreci
### 7.1 Parametre Önem Analizi
![XGBoost HPO Parametre Önemleri](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/b20269d3-4a0a-4e4e-b02a-c7f31ebb6ddf)

*Şekil 11. XGBoost HPO Parametre Önem Analizi — `gamma` (0,39) ve `max_depth` (0,26) en belirleyici parametreler; `learning_rate` (0,12) üçüncü sırada. Düzenlileştirme parametreleri (`reg_alpha`, `reg_lambda`) nispeten düşük önem sergilemiştir.*
### 7.2 Optimizasyon Geçmişi
![XGBoost HPO Optimizasyon Geçmişi](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/d9db5e05-5a8b-463a-81cd-0d7895388213)

*Şekil 12. XGBoost HPO Optimizasyon Geçmişi — En iyi değer 150 trial boyunca kademeli olarak 0,879'a ulaşmıştır; en iyi trial #138'de elde edilmiştir.*
### 7.3 Slice Plot
![XGBoost HPO Slice Plot](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/38aa4dba-3be3-45c8-a683-50eac99eb7bd)

*Şekil 13. XGBoost Slice Plot — Her hiperparametrenin arama uzayı boyunca nesne değerine etkisi.*
### 7.4 Paralel Koordinat Grafiği
![XGBoost HPO Paralel Koordinat](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/f875a8e5-a2f6-47c5-859f-1939f597b40e)

*Şekil 14. XGBoost Paralel Koordinat Grafiği — `max_depth` ve `gamma` kombinasyonlarının performans üzerindeki belirleyici etkisi görülmektedir.*
### 7.5 En İyi Hiperparametre Konfigürasyonu
| Parametre | Değer |
|---|---|
| `n_estimators` | 428 |
| `max_depth` | 8 |
| `learning_rate` | 0,0966 |
| `subsample` | 0,4332 |
| `colsample_bytree` | 0,6743 |
| `reg_lambda` | 0,0482 |
| `reg_alpha` | 0,0034 |
| `gamma` | 0,0037 |
| `min_child_weight` | 4 |
| `max_delta_step` | 14 |
| `image_size` | 224 |
| `class_balance` | balanced |
| **val_f1_mean (3-fold)** | **0,8787** |
| **val_f1_std** | **±0,0085** |
| **best_iteration** | 427 |

***
## 8. XGBoost — Eğitim Süreci
### 8.1 Eğitim Eğrisi ve Özellik Önemi
![XGBoost Eğitim Eğrisi](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/283b9acb-2b29-439a-8917-52fd3e9cef29)

*Şekil 15. XGBoost Eğitim Eğrisi — Validation multi-log loss ve makro F1 loss, boosting round sayısı arttıkça düzenli biçimde azalmaktadır. Model 427. boosting round'da en iyi değerine ulaşmıştır.*

![XGBoost Feature Importance](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/fa4dd823-4221-4932-b53e-4072d2feb40d)

*Şekil 16. XGBoost Özellik Önemi — HOG grubunun toplam önem skorunun %80'inden fazlasını oluşturduğu görülmektedir; GLCM ve LBP tamamlayıcı katkı sunmaktadır.*

***
## 9. XGBoost — Test Sonuçları
### 9.1 Karmaşıklık Matrisi
![XGBoost Confusion Matrix](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/9036398b-88a9-45ae-a74c-2757a0c9721b)

*Şekil 17. XGBoost Karmaşıklık Matrisi — En çok karışım NonDemented ↔ VeryMildDemented arasında gözlemlenmiştir.*

![XGBoost Normalized Confusion Matrix](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/73a49750-177f-4b61-b049-4e6aa646d06c)

*Şekil 18. XGBoost Normalize Karmaşıklık Matrisi — NonDemented %94, VeryMildDemented %94, MildDemented %91, ModerateDemented %100.*

| Sınıf | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NonDemented | 0,965 | 0,941 | 0,953 | 622 |
| VeryMildDemented | 0,870 | 0,941 | 0,904 | 270 |
| MildDemented | 0,955 | 0,914 | 0,934 | 163 |
| ModerateDemented | 0,909 | 1,000 | 0,952 | 10 |
| **Makro Ortalama** | **0,925** | **0,949** | **0,936** | **1.065** |

### 9.2 ROC ve Precision-Recall Eğrileri
![XGBoost ROC PR Curves](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/0ecfb997-b882-4bb8-aace-afaf15a232cb)

*Şekil 19. XGBoost One-vs-Rest ROC ve Precision-Recall Eğrileri.*

| Metrik | Değer |
|---|---:|
| **Makro AUC-OVR** | **0,9920** |
| **Makro AP** | **0,9850** |

***
## 10. Model Karşılaştırması
### 10.1 Performans Metrikleri
| Metrik | ResNet18 (Tuned) | XGBoost (Tuned) |
|---|---:|---:|
| Test Doğruluğu | **%98,87** | %93,71 |
| Makro F1 (Test) | **0,9912** | 0,9358 |
| Makro F1 (CV, HPO) | **0,9797 ± 0,0034** | 0,8787 ± 0,0085 |
| Makro AUC-OVR | **0,9998** | 0,9920 |
| Makro AP | **0,9995** | 0,9850 |
| NonDemented F1 | **0,990** | 0,953 |
| VeryMild F1 | **0,984** | 0,904 |
| Mild F1 | **0,991** | 0,934 |
| Moderate F1 | **1,000** | 0,952 |
| Ortalama tahmin güveni | **0,955** | 0,863 |
| Yüksek güvenli hata sayısı | **3** | 2 |
### 10.2 Literatürle Karşılaştırma
| Çalışma | Model | Veri | Doğruluk / F1 |
|---|---|---|---|
| Kaur ve ark. (2025)[^1] | ResNet-50 + EfficientNet-B3 (Ensemble) | 33.984 (augmented) | %99,32 |
| Kırtay & Koçak (2024)[^6] | ResNet50 + SVM | 6.400 (orijinal) | %98 |
| **Bu çalışma — ResNet18** | **ResNet18 (Transfer Learning)** | **33.984 (augmente)** | **%98,87 / F1=0,991** |
| PMC (2025)[^4] | InceptionResNetV2 | 6.735 görüntü | %99 |
| **Bu çalışma — XGBoost** | **XGBoost (HOG+LBP+GLCM, class-balanced)** | **33.984 (augmente)** | **F1=0,936 (test)** |
| Openbio. (2022)[^5] | XGBoost (morfoloji özellikleri) | ADNI | %92,31 |

ResNet18, aynı augmente veri seti üzerinde Kaur ve ark. (2025)'ın ResNet-50 + EfficientNet-B3 topluluğuna (%99,32) yakın performans (%98,87) sergilemiştir. Bu sonuç, tek bir hafif mimarinin iyi ayarlanmış transfer öğrenme stratejisiyle karmaşık topluluklara rekabetçi bir alternatif oluşturabileceğini göstermektedir.[^8][^9]
### 10.3 Sınıf Karışım Örüntüleri
ResNet18, XGBoost'un en çok zorlandığı NonDemented ↔ VeryMildDemented sınır bölgesinde belirgin bir iyileşme sağlamıştır. XGBoost'ta NonDemented için yaklaşık %6 yanlış sınıflandırma oranı ResNet18'de %2'ye, VeryMildDemented için %6 oranı ise %0'a inmiştir. Bu iyileşme, uçtan uca öğrenilen özelliklerin komşu demans evreleri arasındaki nüanslı görsel ayrımları el yapımı özelliklerden daha iyi yakaladığına işaret etmektedir.[^4][^17]

ResNet18, ModerateDemented sınıfında mükemmel sınıflandırma (F1=1,000) gösterirken XGBoost bu sınıfta F1 = 0,952 elde etmiştir; bununla birlikte XGBoost'ta da recall = 1,000 olarak gözlemlenmiştir (yanlış negatif = 0). İleri evre beyin değişikliklerinin — belirgin kortikal atrofi ve ventriküler genişleme — her iki model tarafından yüksek duyarlılıkla ayrıştırılabildiği görülmektedir.[^5]

***
## 11. Tartışma
### 11.1 Derin Öğrenmenin Üstünlüğü
ResNet18'in XGBoost karşısında tüm metriklerde belirgin üstünlük sergilemesi, CNN'lerin hiyerarşik özellik öğrenimi kapasitesiyle açıklanmaktadır. Alt konvolüsyonel katmanlar kenar ve doku gibi düşük düzeyli özellikleri, üst katmanlar ise Alzheimer'a özgü anatomik örüntüleri — hippokampal atrofi, sulcal gidişat, gri madde kaybı — öğrenmektedir. El yapımı HOG/LBP/GLCM özellikleri bu hiyerarşik temsilleri yakalamakta yetersiz kalmaktadır.[^4][^1]

Transfer öğrenme stratejisinin etkinliği de bu çalışmada teyit edilmiştir. ImageNet ağırlıklarıyla başlatılan ResNet18, yalnızca 23 epoch eğitimle %98,87 test doğruluğuna ulaşmış; bu durum sınırlı eğitim süresiyle yüksek performans elde edilebileceğini göstermektedir.[^7][^18]
### 11.2 XGBoost'un Tamamlayıcı Değeri
Test setinde makro F1 = 0,936 değeriyle XGBoost, klinisyenler için yorumlanabilirlik gerektiren senaryolarda değerli bir tamamlayıcı model olma niteliği taşımaktadır. Özellik önemi analizi, HOG özelliklerinin demans sınıflandırmasında baskın bilgi kaynağı olduğunu ortaya koymuştur. Bu bulgu, beyin morfolojisini temsil eden yönelim gradyanlarının — farklı demans evrelerinde farklılaşan gri madde kayıplarını yansıtan — HOG ile etkili biçimde kodlandığını göstermektedir. Geleneksel ML modellerinin çıkarım süresi ve kaynak kullanımı açısından derin öğrenme modellerine kıyasla belirgin avantajlar sunduğu da unutulmamalıdır.
### 11.3 HPO Stratejileri Karşılaştırması
ResNet18 için `lr` ve `dropout`, XGBoost için ise `gamma` ve `max_depth` en kritik hiperparametreler olarak öne çıkmıştır. Bu fark, iki modelin optimizasyon manzarasındaki temel ayrışmayı yansıtmaktadır: derin öğrenmede öğrenme dinamikleri (öğrenme hızı, regularizasyon), gradyan artırmada ise ağaç karmaşıklığı ve cezalandırma mekanizmaları belirleyici rol oynamaktadır.[^16][^19]

***
## 12. Kısıtlamalar ve Gelecek Çalışmalar
- **2D dilim kısıtı:** Her iki model de 3D volumetrik MRI bilgisini kullanmamaktadır; 3D yaklaşımlar anatomik bağlamı daha kapsamlı biçimde temsil edebilir.[^20]
- **Test seti boyutu:** ModerateDemented için test seti yalnızca 10 örnek içermektedir; bu sınıftaki mükemmel sonuçların istatistiksel güvenilirliği sınırlıdır.
- **Harici doğrulama:** Bağımsız bir klinik kohortla prospektif değerlendirme gerçekleştirilmemiştir.
- **Açıklanabilirlik:** ResNet18 için Grad-CAM görselleştirmesi, modelin hangi beyin bölgelerine odaklandığını klinisyenler için görünür kılacak ve klinik güveni artıracaktır.[^21][^22]

Gelecek çalışmalar için önerilen yönelimler: ResNet18 + XGBoost topluluk modeli, Grad-CAM entegrasyonu, çok modlu veri (görüntü + klinik veri) bütünleştirilmesi ve daha büyük test kohortlarıyla dış doğrulama.[^23]

***
## 13. Sonuç
Bu çalışmada 33.984 augmente beyin MRI görüntüsü üzerinde ResNet18 ve XGBoost modelleri karşılaştırılmıştır. ResNet18, Optuna Bayesian HPO ile optimize edilerek test setinde %98,87 doğruluk ve makro F1 = 0,991 değerlerine ulaşmış; tüm sınıflarda AUC ≥ 0,999 elde etmiştir. XGBoost ise HOG/LBP/GLCM tabanlı 4.451 boyutlu özellik vektörü, sınıf ağırlıklandırması (balanced) ve 3-fold CV HPO ile test setinde makro F1 = 0,936 ve doğruluk = %93,71 değerlerini yakalamıştır. ResNet18, ModerateDemented sınıfında F1 = 1,000 ile mükemmel sınıflandırma gerçekleştirirken XGBoost bu sınıfta F1 = 0,952 ve recall = 1,000 elde etmiştir. Sızıntısız deneysel tasarım ve kaynak grup bazlı bölme stratejisi, elde edilen performans değerlerinin gerçek genelleme kapasitesini güvenilir biçimde yansıttığını güvence altına almaktadır.[^24]

---

## References

1. [Intelligent Alzheimer's diagnosis and disability assessment](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1619228/full) - The dataset used in this study is a publicly available MRI dataset sourced from Kaggle, titled the “...

2. [World Alzheimer Report 2024](https://www.alzint.org/u/World-Alzheimer-Report-2024.pdf) - World Alzheimer Report 2015 – The Global Impact of Dementia: An analysis of prevalence, incidence, c...

3. [Dementia](https://www.who.int/health-topics/dementia) - In 2021, 57 million people worldwide lived with dementia, with over 60% in low- and middle-income co...

4. [Classifying and diagnosing Alzheimer's disease with deep ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12216300/) - This study aims to utilize deep convolutional neural networks (CNNs) trained on MRI data for Alzheim...

5. [High Accuracy Diagnosis for MRI Imaging Of Alzheimer's ...](https://openbiotechnologyjournal.com/VOLUME/16/ELOCATOR/e187407072208300/FULLTEXT/) - By implementing XGBoost for the selected 16 features of the four groups of MRI images, the classific...

6. [TRANSFER LEARNING IN SEVERITY CLASSIFICATION ...](https://dergipark.org.tr/en/download/article-file/3386757) - Utilizing brain MRI images to classify dementia stages, our experimental analysis revealed that tran...

7. [A transfer learning approach for multiclass classification of ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC9869687/) - In this study, we propose a transfer learning base approach to classify various stages of AD. The pr...

8. [ResNet-18](https://aicuflow.com/docs/tool/training/computer_vision/image_classification/resnet-18) - ResNet-18 is the smallest variant of the Residual Network family, featuring 18 layers with skip conn...

9. [Comparative Analysis of Lightweight Deep Learning ...](https://arxiv.org/html/2505.03303v1) - ResNet18: Delivered strong accuracy (96.05%) and F1 score (0.9578), with the fastest inference time ...

10. [Data Augmentation for Brain-Tumor Segmentation: A Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC6917660/) - In this paper, we review the current advances in data-augmentation techniques applied to magnetic re...

11. [Imbalance-aware loss functions improve medical image ...](https://openreview.net/forum?id=5Oiqw76ube) - In this study, we aim to improve medical image classification by effectively addressing class imbala...

12. [Enhancing early detection of Alzheimer's disease through ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC11682306/) - Thus, the CLAHE method helps improve the accuracy of AD diagnosis by increasing the visibility of br...

13. [A Survey on Detection of Alzheimer's Disease from Brain ...](https://ijirt.org/publishedpaper/IJIRT177962_PAPER.pdf) - The proposed system enhances Alzheimer's diagnosis by applying advanced preprocessing techniques lik...

14. [Machine learning-driven GLCM analysis of structural MRI for ...](https://ciencia.ucp.pt/en/publications/machine-learning-driven-glcm-analysis-of-structural-mri-for-alzhe/) - ## Abstract

15. [Brain tumor classification: a novel approach integrating GLCM ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC10861642/) - The proposed Composite Feature Extraction model utilizing GLCM, LBP, and Composite Features achieves...

16. [Optuna: A hyperparameter optimization framework — Optuna ...](https://optuna.readthedocs.io) - Optuna is an automatic hyperparameter optimization software framework, particularly designed for mac...

17. [Using ResNet-18 in a deep-learning framework and ...](https://opus.bibliothek.uni-augsburg.de/opus4/files/112606/upload_files_doi-10.5455-jjcit.71-16998184061708842869-33.pdf) - The entire process encompasses image preprocessing, classification and performance assessment [14]. ...

18. [transfer learning with ResNet50 for MRI-based diagnosis](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1664418/full) - The results confirm that transfer learning using ResNet50 significantly enhances the accuracy and sc...

19. [Combining K-fold cross validation with bayesian ...](https://www.nature.com/articles/s41598-025-23336-w) - This improvement in overall accuracy demonstrates the effectiveness of combining Bayesian hyperparam...

20. [3D Brain MRI Classification for Alzheimer's Diagnosis ...](https://arxiv.org/html/2505.04097v1) - This study proposes a three-dimensional deep learning model (3D CNN) for classifying brain magnetic ...

21. [Early diagnosis of Alzheimer's Disease using hybrid CNN- ...](https://dergipark.org.tr/en/download/article-file/4938536) - Grad-CAM plays a critical role in enhancing the interpretability of DL models by highlighting the sp...

22. [Deep learning for Alzheimer's disease: advances in ... - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12752331/) - Explainable AI (XAI) techniques like Grad-CAM, Integrated Gradients, SHAP and LIME are increasingly ...

23. [Alzheimer disease predicting from clinical and MRI data ...](https://www.nature.com/articles/s41598-025-28221-0) - The distribution of samples across classes in the MRI dataset is as follows: Mild Demented (approx. ...

24. [A Guide to Cross-Validation for Artificial Intelligence in ... - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10388213/) - In k-fold CV, the dataset is partitioned patientwise into k disjoint sets called folds (Fig 3). Firs...

