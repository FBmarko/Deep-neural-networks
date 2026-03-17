# Ödev-2: CNN Katmanları Matematiksel Analizi (Slayt 22 ve 29)

Bu belgede, `DL_L5_CNNs.pptx` sunumunun 22. ve 29. slaytlarındaki matematiksel kavramların detaylı analizi yer almaktadır.

---

## 1. Evrişim (Convolution) Katmanı Boyut Hesaplaması (Slayt 22)

**Slayttaki Veriler:** 
- `Giriş (Input) = 5x5x3`
- `Padding (P) = 1`
- `Stride (S) = 2`

Bu slaytta arka planda çalışan ve bir görüntü matrisi evrişim işleminden çıktıktan sonra oluşan **yeni (çıkış) boyutunu hesaplayan matematiksel formül** işlenmektedir. Formül şu şekildedir:

> **O = [(W - F + 2P) / S] + 1**

- **W (Genişlik/Yükseklik):** Giriş resminin boyutu (Slaytta girişi 5x5 verilmiş, yani W = 5).
- **F (Filtre/Kernel Boyutu):** Slaytta spesifik bir filtre boyutu yazmamakla beraber, girdi boyutuna göre genelde 3x3 kullanılır (F = 3 kabul edelim).
- **P (Padding/Dolgu):** Resmin kenarlarına eklenen 0'ların katmanı. Slaytta P = 1.
- **S (Stride/Adım):** Filtrenin resim üzerinde kaçar kaçar kaydırılacağı. Slaytta S = 2.

**Uygulaması:** 
Formülü slayttaki (5x5, P=1, S=2) ve standart bir filtreyle (3x3) uygularsak:
1. Pay kısmı: `(5 - 3 + 2 * 1) = 4`
2. Bölme: `4 / 2 (Stride)` = 2
3. Sabit toplama: `2 + 1` = **3**

**Sonuç:** Çıkış matrisimiz `3x3` boyutlarına düşecektir. Padding (1) eklendiği için köşelerdeki veriler de işlenmiş ancak Stride (2) adımı büyük seçildiği için genel boyut 5'ten 3'e küçültülmüştür. Bu azaltma işlemi, özellik çıkarımı (feature extraction) yaparken veri yükünü büyük ölçüde hafifletir.

---

## 2. Softmax Fonksiyonu Matematiği (Slayt 29)

**Slayttaki Bilgi:** *"Softmax fonksiyonu sınıflandırma problemlerinde kullanılır. Softmax katmanı çıkış sınıflarının olasılık dağılımını hesaplar."*

Softmax, sinir ağının en son (çıkış) katmanında bulunan, ağın ürettiği ham ve anlamsız skorları (Logits) alıp her bir sınıf için **0 ile 1 arasında olasılıklara (% yüzdelere)** dönüştüren aktivasyon fonksiyonudur.

**Matematiksel Formülü:**

> $$Softmax(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

- **$z_i$**: İlgili sınıf için ağın ürettiği ham skor (logit).
- **$e^{z_i}$**: Logit değerinin Euler sayısı ($e \approx 2.718$) tabanında üstel fonksiyonu alınır. Bunun amacı, negatif skorları eksiden kurtarıp pozitife, zaten büyük olan pozitif skorları ise daha da çarpıcı hale getirerek öne çıkarmaktır.
- **$\sum e^{z_j}$**: Ağdan çıkan tüm üstel değerlerin toplamıdır (Payda).

**Nasıl Çalışır? (Örnekle):**
Ağımız bir nesneyi; Kedi, Köpek ve Kuş olarak tahminlemeye (sınıflandırmaya) çalışsın.
Ham skorlar (Logits) şu şekilde gelmiş olsun: *Kedi= 2.0, Köpek= 1.0, Kuş= 0.1*.

1. Öncelikle hepsinin `$e$` tabanında üssü alınır: 
   - $e^{2.0} = 7.39$
   - $e^{1.0} = 2.71$
   - $e^{0.1} = 1.10$
2. Bunların toplamı (Payda) hesaplanır: `7.39 + 2.71 + 1.10 = 11.20`.
3. Her bir üstel değer, elde edilen bu toplama bölünerek olasılığa çevrilir:
   - **Kedi:** `7.39 / 11.20 = 0.66` (Yani %66 Olasılık)
   - **Köpek:** `2.71 / 11.20 = 0.24` (Yani %24 Olasılık)
   - **Kuş:** `1.10 / 11.20 = 0.10` (Yani %10 Olasılık)

**Matematiğinin Avantajı:** 
Softmax, çıkan tüm tahmin olasılıklarının toplamının daima %100 (1.0) olmasını kesin olarak garanti eder. Ayrıca üstel ($e^x$) fonksiyon kullandığı için ağın biraz yüksek puan verdiği en olası skoru "dominant" hale getirerek öne çıkarırken, negatif skorları 0'a çok daha fazla yaklaştırır. Böylece Cross-Entropy gibi loss fonksiyonları ile kayıp hesabı (hata hesaplaması) çok stabil ve tutarlı hale gelir. 
