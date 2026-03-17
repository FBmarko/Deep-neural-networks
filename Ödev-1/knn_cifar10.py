import numpy as np
import pickle
import os


dataset_path = 'cifar-10-batches-py'

if not os.path.exists(dataset_path):
    print(f"HATA: '{dataset_path}' klasörü bulunamadı!")
    print("Veriyi internetten çekmiyoruz. Lütfen veri setini indirip 'cifar-10-batches-py' klasörünü buraya koyun.")
    exit()

print("CIFAR-10 veri seti okunuyor...")

with open(os.path.join(dataset_path, 'data_batch_1'), 'rb') as f_train:
    train_data_dict = pickle.load(f_train, encoding='bytes')
    X_train = train_data_dict[b'data']
    Y_train = np.array(train_data_dict[b'labels'])


with open(os.path.join(dataset_path, 'test_batch'), 'rb') as f_test:
    test_data_dict = pickle.load(f_test, encoding='bytes')
    X_test = test_data_dict[b'data']
    Y_test = np.array(test_data_dict[b'labels'])

with open(os.path.join(dataset_path, 'batches.meta'), 'rb') as f_meta:
    meta_dict = pickle.load(f_meta, encoding='bytes')
    label_names = [name.decode('utf-8') for name in meta_dict[b'label_names']]

print(f"Eğitim Verisi: {X_train.shape[0]} adet görsel yüklendi.")
print(f"Test Verisi: {X_test.shape[0]} adet görsel yüklendi.\n")


mesafe_secimi = ""
while True:
    mesafe_secimi = input("Hangi mesafe metriği kullanılsın? (L1 için 'L1', L2 için 'L2' yazın): ").strip().upper()
    if mesafe_secimi in ['L1', 'L2']:
        break
    print("Lütfen sadece L1 veya L2 girin.\n")

k_degeri = 0
while True:
    try:
        k_degeri = int(input("Komşu sayısı (k) değerini giriniz: "))
        if k_degeri > 0:
            break
        print("K değeri 0'dan büyük olmalıdır.\n")
    except ValueError:
        print("Lütfen geçerli bir sayı girin.\n")



indeks = 0 
test_resmi = X_test[indeks]
gercek_etiket = Y_test[indeks]
gercek_sinif_adi = label_names[gercek_etiket]

print("\n" + "="*40)
print(f"Sınıflandırma Başlıyor...")
print(f"- Seçilen Test Resmi Sırası: {indeks}")
print(f"- Test Resminin Gerçek Sınıfı: {gercek_sinif_adi}")
print(f"- Seçilen Metrik: {mesafe_secimi}")
print(f"- Seçilen k Değeri: {k_degeri}")
print("="*40)

mesafeler = []


if mesafe_secimi == 'L1':

    farklar = np.abs(X_train - test_resmi)
    mesafeler = np.sum(farklar, axis=1)
elif mesafe_secimi == 'L2':

    fark_kareleri = np.square(X_train - test_resmi)
    mesafeler = np.sqrt(np.sum(fark_kareleri, axis=1))


sirali_indeksler = np.argsort(mesafeler)
en_yakin_k_indeksler = sirali_indeksler[:k_degeri]


oy_listesi = []
for i in en_yakin_k_indeksler:
    oy_listesi.append(Y_train[i])

etiket_sayilari = {}
for oy in oy_listesi:
    if oy in etiket_sayilari:
        etiket_sayilari[oy] += 1
    else:
        etiket_sayilari[oy] = 1


tahmin_edilen_etiket = None
en_yuksek_oy = -1

for etiket, oy_sayisi in etiket_sayilari.items():
    if oy_sayisi > en_yuksek_oy:
        en_yuksek_oy = oy_sayisi
        tahmin_edilen_etiket = etiket

tahmin_edilen_sinif_adi = label_names[tahmin_edilen_etiket]

print(f"\n=> Seçilen k= {k_degeri} için Yakın Komşuların Sınıfları:")
for idx, i in enumerate(en_yakin_k_indeksler):
    ksinif = label_names[Y_train[i]]
    kmesafe = mesafeler[i]
    print(f"  {idx+1}. Komşu Sınıfı: {ksinif} (Mesafe: {kmesafe:.2f})")

print("-"*40)
print(f"==> TAHMİN EDİLEN SINIF: {tahmin_edilen_sinif_adi}")
print(f"==> GERÇEK SINIF: {gercek_sinif_adi}")

if tahmin_edilen_etiket == gercek_etiket:
    print("\nSONUÇ: Başarılı Tahmin! 🎉")
else:
    print("\nSONUÇ: Yanlış Tahmin. 😢")
