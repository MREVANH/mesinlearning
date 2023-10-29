# Laporan Proyek Machine Learning
### Nama : MOCHAMMAD REVAN H
### NIM : 211351084
### KELAS : IF MALAM A

## DOMAIN PROYEK

Proyek ini bertujuan untuk mengembangkan sebuah sistem manajemen penjualan mobil berbasis web yang efisien dan komprehensif. Sistem ini akan memungkinkan dealer mobil atau pemilik mobil untuk mengelola stok mereka, berinteraksi dengan pelanggan, mengelola transaksi, dan melacak inventaris dengan lebih baik. Beberapa fitur utama yang akan diintegrasikan dalam proyek ini termasuk:

## BUSINESS UNDERSTANDING

Proyek ini bertujuan untuk proses data mining dan analisis data yang bertujuan untuk memahami secara mendalam konteks bisnis dan masalah yang ingin dipecahkan. Dalam konteks "Jual Mobil

### PROBLEM STATEMENTS

Problem Statements untuk proyek dengan judul "Jual Mobil" dapat berfokus pada sejumlah tantangan yang sering dihadapi dalam bisnis penjualan mobil. Berikut ini adalah beberapa contoh Problem Statements:

1. Harga yang Tidak Optimal: Penentuan harga yang tepat dan kompetitif untuk mobil yang dijual adalah tantangan, dan harga yang terlalu tinggi atau terlalu rendah dapat mempengaruhi profitabilitas.

2. Kesulitan dalam Menilai Mobil Bekas: Penilaian yang akurat tentang nilai mobil bekas adalah penting, tetapi sering kali sulit dilakukan
3. Penyusutan Nilai Mobil: Mobil yang lama berada dalam stok sering mengalami penurunan nilai, yang dapat mengurangi keuntungan.

### GOALS

 Tujuan utama dapat difokuskan dalam proyek "Jual Mobil" untuk meningkatkan bisnis penjualan mobil:

1. Meningkatkan Penjualan: Tujuan utama adalah meningkatkan jumlah mobil yang dijual dengan cara yang menguntungkan.

2. Peningkatan Pelayanan Pelanggan: Memastikan pelayanan pelanggan yang lebih baik, termasuk pengurangan waktu tunggu pelanggan, respon yang cepat terhadap pertanyaan, dan peningkatan kepuasan pelanggan

3. Konservasi Nilai Mobil: Meminimalkan depresiasi nilai mobil dalam stok.

### SOLUTIONS STATEMENTS

Solusi Statement adalah pernyataan yang merinci solusi atau pendekatan yang akan diambil untuk mengatasi Problem Statements yang telah diidentifikasi dalam proyek "Jual Mobil." Berikut adalah contoh Solusi Statement yang mungkin diterapkan dalam proyek ini:

1. Optimasi Harga: Kami akan melakukan analisis harga kompetitif dan menentukan strategi harga yang optimal berdasarkan data dan pemahaman pelanggan.

2. Penilaian yang Akurat terhadap Mobil Bekas: Kami akan mengembangkan algoritma penilaian otomatis yang mengintegrasikan data kondisi mobil dan data pasar untuk menilai nilai mobil bekas.

3. Penggunaan Teknologi Inovatif: Kami akan mengeksplorasi penggunaan teknologi baru seperti teknologi VR/AR, blockchain, dan aplikasi seluler untuk memperbaiki pengalaman pelanggan dan efisiensi operasional.

## DATA UNDERSTADING

Dari dataset yang kita dapatkan dari kaggle

(https://www.kaggle.com/datasets/mahmudulislamovi/car-price-prediction)

### Variabel-variabel pada Car Price Prediction Dataset adalah sebagai berikut:

 0   Manufacturer            object 
 1   Model                   object 
 2   Sales_in_thousands      float64
 3   __year_resale_value     float64
 4   Vehicle_type            object 
 5   Price_in_thousands      float64
 6   Engine_size             float64
 7   Horsepower              float64
 8   Wheelbase               float64
 9   Width                   float64
 10  Length                  float64
 11  Curb_weight             float64
 12  Fuel_capacity           float64
 13  Fuel_efficiency         float64
 14  Latest_Launch           object 
 15  Power_perf_factor       float64

## DATA PREPARATION

## DATA COLLECTION

Data ini saya dapatkan dari kaggle tentang Car Price Prediction Dataset

## DATA DISCOVERY AND PROFILING

#### - Mengimport library yang di butuhkan

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#### - Memanggil Dataset yang digunakan

df = pd.read_csv('Car_sales.csv')

#### - Melihat 5 data yang dipanggil

df.head()

#### - Selanjutnya melihat info dari data 

df.info()

#### - Mencari Heatmap

sns.heatmap(df.isnull()) digunakan untuk membuat sebuah heatmap yang menggambarkan lokasi dan melihat data yang hilang dalam sebuah dataframe

Penjelasan komponen-komponen utama dari kode tersebut adalah sebagai berikut:

1. `sns`: Ini adalah singkatan dari Seaborn, yang merupakan library Python yang sering digunakan untuk membuat visualisasi data yang lebih menarik dan informatif.

2. `.heatmap()`: Ini adalah metode dalam Seaborn yang digunakan untuk membuat visualisasi peta panas. Metode ini menerima input berupa matriks data, dan akan menghasilkan tampilan warna yang memperlihatkan nilai-nilai dalam matriks tersebut.

3. `df.isnull()`: Ini adalah ekspresi yang digunakan untuk mengidentifikasi data yang hilang (missing data) dalam DataFrame `df`. Ketika dipanggil pada DataFrame, metode `.isnull()` mengembalikan DataFrame yang memiliki nilai `True` di tempat di mana data hilang, dan `False` di tempat di mana data tersedia.

``` bash
sns.heatmap(df.isnull())
```

#### - Disini kita akan melihat nilai data

``` bash
df.describe()
```

#### - Ini adalah bagian visualisasi data 
1. plt.figure(figsize=(1,2)): Ini adalah baris pertama dalam kode. Ini digunakan untuk membuat sebuah gambar atau figur dengan ukuran yang ditentukan. Dalam kasus ini, gambar akan memiliki lebar 10 inci dan tinggi 6 inci, sesuai dengan parameter (1, 2) yang diberikan. Ukuran gambar ini berguna untuk mengendalikan seberapa besar atau kecil gambar heatmap akan muncul saat dihasilkan.

2. sns.heatmap(df.corr(), annot=True): Baris ini digunakan untuk membuat heatmap dari korelasi antar variabel dalam DataFrame df dengan menggunakan library Seaborn (sns adalah alias untuk Seaborn). Ini terdiri dari dua bagian utama:

- df.corr(): Ini adalah panggilan fungsi corr() pada DataFrame df, yang menghitung matriks korelasi antar variabel dalam dataset. Matriks korelasi adalah sebuah matriks yang menunjukkan sejauh mana dua variabel berhubungan satu sama lain. Nilai korelasi berkisar antara -1 (korelasi negatif sempurna) hingga 1 (korelasi positif sempurna). Nilai 0 menunjukkan tidak adanya korelasi. Hasil dari df.corr() adalah matriks korelasi ini.

- annot=True: Parameter ini digunakan untuk menambahkan angka-angka ke dalam sel-sel heatmap. Angka-angka ini adalah nilai korelasi antar variabel yang ditunjukkan oleh warna di heatmap. Jika annot diatur sebagai True, maka angka-angka tersebut akan ditampilkan di setiap sel heatmap.

``` bash
plt.figure(figsize=(1,2))
sns.heatmap(df.corr(), annot=True)
```
![Alt text](gmbr2.png) <br>

- Code program `df.isnull().sum()` digunakan untuk menghitung jumlah nilai-nilai yang hilang (NaN/null) dalam sebuah DataFrame (biasanya dinotasikan sebagai `df`), dan mengembalikan hasil dalam bentuk sebuah Series yang menunjukkan jumlah nilai-nilai yang hilang untuk setiap kolom dalam DataFrame.

Ini adalah beberapa penjelasan untuk setiap bagian dari kode program tersebut:

1. `df`: Ini adalah DataFrame yang akan dianalisis. Sebuah DataFrame adalah struktur data dalam bahasa pemrograman Python yang sering digunakan untuk menyimpan dan mengelola data dalam bentuk tabel, seperti spreadsheet. Dalam hal ini, `df` adalah nama yang digunakan untuk merujuk pada DataFrame tersebut.

2. `.isnull()`: Ini adalah metode yang digunakan pada DataFrame yang akan mengembalikan DataFrame baru dengan nilai-nilai boolean (True/False) yang menunjukkan apakah setiap elemen dalam DataFrame adalah null (NaN) atau tidak. Hasil dari `.isnull()` adalah DataFrame baru yang memiliki bentuk yang sama dengan `df`, tetapi berisi True jika elemen adalah null dan False jika tidak.

3. `.sum()`: Ini adalah metode lain yang digunakan pada DataFrame, dan dalam konteks ini, ia menghitung jumlah True dalam setiap kolom DataFrame yang dihasilkan oleh `.isnull()`. Ini mengembalikan hasil dalam bentuk Series yang menunjukkan jumlah nilai-nilai yang hilang (True) dalam setiap kolom DataFrame.

``` bash
df.isnull().sum()
```

#### - MODELING DATA

Kode program di bawah ini terlihat seperti mengambil dua kolom dari suatu DataFrame (df) sebagai input fitur (x) dan satu kolom sebagai target (y):

- Pada baris pertama, `x` adalah variabel yang akan menyimpan fitur-fitur dari dataset. Dua kolom yang diambil dari DataFrame adalah 'housing_median_age' dan 'ocean_proximity'. Biasanya, dalam pemrosesan data, kita memilih kolom-kolom tertentu yang dianggap relevan atau penting untuk analisis atau pemodelan. Kolom 'housing_median_age' berisi data tentang usia median perumahan, dan 'ocean_proximity' berisi informasi tentang kedekatan properti dengan garis pantai atau perairan (mungkin dalam bentuk kategori seperti "NEAR BAY", "INLAND", dll.). Ini adalah fitur-fitur yang akan digunakan dalam analisis atau pemodelan lebih lanjut.

- Pada baris kedua, `y` adalah variabel yang akan menyimpan target dari dataset. Dalam hal ini, kolom 'latitude' dipilih sebagai target. Ini mungkin berarti bahwa tujuan dari analisis atau pemodelan ini adalah untuk memprediksi atau menggambarkan informasi yang terkait dengan data latitude berdasarkan fitur-fitur yang ada dalam `x`. Latitude adalah koordinat geografis yang mengacu pada lokasi geografis suatu properti, dan kemungkinan besar digunakan sebagai variabel target dalam analisis spasial atau pemodelan geografis.

Selanjutnya, Anda dapat menggunakan `x` dan `y` ini untuk melakukan berbagai analisis data atau membangun model statistik atau machine learning, tergantung pada tujuan dan konteks dari proyek atau analisis yang sedang dilakukan.
``` bash


#### - Split Data & Testing

Kode program di bawah menggunakan library `scikit-learn` (sklearn) untuk membagi dataset menjadi data pelatihan (training data) dan data pengujian (testing data). Biasanya, kita memiliki dataset yang terdiri dari dua komponen: fitur (features) yang disimpan dalam variabel `X` dan target atau label yang disimpan dalam variabel `y`. Tujuan utama dari kode ini adalah untuk memisahkan dataset ini menjadi dua bagian: satu untuk melatih model (X_train dan y_train) dan satu untuk menguji model (X_test dan y_test).

Di sini adalah penjelasan mengenai kode tersebut:

1. `from sklearn.model_selection import train_test_split`: Pada baris pertama kode, kita mengimpor fungsi `train_test_split` dari modul `model_selection` dalam library `scikit-learn`. Fungsi ini digunakan untuk membagi dataset menjadi dua subset: data pelatihan dan data pengujian.

2. `X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)`: Pada baris kedua kode, kita menggunakan `train_test_split` untuk membagi dataset. Parameter yang digunakan adalah sebagai berikut:
- `x`: Ini adalah variabel yang berisi fitur atau atribut dari dataset.
- `y`: Ini adalah variabel yang berisi label atau target yang ingin diprediksi.
- `test_size`: Parameter ini menentukan seberapa besar bagian dataset yang akan digunakan untuk data pengujian. Dalam contoh di atas, `test_size=0.2` berarti bahwa 20% dari dataset akan digunakan sebagai data pengujian, sedangkan 80% sisanya akan digunakan sebagai data pelatihan.

Hasil dari pemanggilan `train_test_split` ini adalah empat variabel:
- `X_train`: Ini berisi subset dari fitur (X) yang akan digunakan sebagai data pelatihan.
- `X_test`: Ini berisi subset dari fitur (X) yang akan digunakan sebagai data pengujian.
- `y_train`: Ini berisi subset dari label (y) yang sesuai dengan data pelatihan.
- `y_test`: Ini berisi subset dari label (y) yang sesuai dengan data pengujian.

Dengan memisahkan dataset menjadi data pelatihan dan data pengujian, kita dapat melatih model pada data pelatihan dan menguji kinerja model tersebut pada data pengujian. Ini adalah praktik umum dalam pembuatan model machine learning untuk mengukur sejauh mana model tersebut mampu melakukan prediksi yang baik pada data yang belum pernah dilihat sebelumnya (data pengujian).

``` bash
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```
#### - Seleksi Fitur

Kode program di bawah ini tampaknya berhubungan dengan pemilihan fitur (features) dari suatu dataset dan mengatur variabel untuk data input (x) dan target (y) dalam konteks analisis data atau pemodelan statistik/machine learning:

```python
fitur = ['Engine_size', 'Horsepower','Wheelbase','Width','Length','Fuel_capacity','Fuel_efficiency']
```

Pada bagian ini, kita mendefinisikan sebuah daftar (list) bernama `fitur`. Daftar ini berisi nama-nama fitur atau variabel-variabel yang akan digunakan dalam analisis data atau pemodelan. Dalam hal ini, fitur yang akan digunakan adalah 'Engine_size', 'Horsepower','Wheelbase','Width','Length','Fuel_capacity','Fuel_efficiency'. Ini berarti kita akan memfokuskan analisis pada tujuh variabel tersebut.

```python
x = df[fitur]
```

Kode ini mengatur variabel `x` untuk menjadi subset dari dataframe `df` yang hanya berisi kolom-kolom yang sesuai dengan daftar fitur yang telah didefinisikan sebelumnya. Dengan kata lain, `x` akan berisi data input yang terdiri dari kolom 'Engine_size', 'Horsepower','Wheelbase','Width','Length','Fuel_capacity','Fuel_efficiency' dari dataframe `df`.

```python
y = df['Price_in_thousands']
```

Pada bagian ini, variabel `y` diatur untuk menjadi kolom 'households' dari dataframe `df`. Ini menunjukkan bahwa `y` adalah target atau variabel yang akan diprediksi atau dianalisis dalam konteks analisis data atau pemodelan yang sedang dilakukan.

Dengan demikian, keseluruhan kode ini bertujuan untuk memilih sejumlah fitur tertentu (dalam hal ini, 'Engine_size', 'Horsepower','Wheelbase','Width','Length','Fuel_capacity','Fuel_efficiency') dari suatu dataframe (dalam hal ini, `df`) untuk digunakan sebagai data input (x) dan menentukan variabel target (y) yang akan digunakan dalam analisis data atau pemodelan yang sedang dilakukan.

``` bash
fitur = ['Engine_size', 'Horsepower','Wheelbase','Width','Length','Fuel_capacity','Fuel_efficiency']
x = df[fitur]
y = df['Price_in_thousands']
```

- Di bawah ini adalah penjelasan langkah-langkah yang dilakukan oleh kode di bawah:

1. **Import LinearRegression dari sklearn**:
   ```python
   from sklearn.linear_model import LinearRegression
   ```
   Kode ini mengimpor modul LinearRegression dari pustaka scikit-learn. Modul ini berisi implementasi model regresi linear yang akan digunakan dalam analisis.

2. **Inisialisasi objek LinearRegression**:
   ```python
   lr = LinearRegression()
   ```
   Baris ini membuat objek LinearRegression yang disimpan dalam variabel `lr`. Objek ini akan digunakan untuk melatih (fit) model regresi linear dan untuk membuat prediksi berdasarkan model yang telah dilatih.

3. **Melatih model regresi linear**:
   ```python
   lr.fit(X_train, y_train)
   ```
   Di sini, model regresi linear dilatih dengan menggunakan data latih. `X_train` adalah matriks fitur (fitur dari data latih) dan `y_train` adalah vektor target (label yang sesuai dengan data latih). Proses pelatihan ini adalah saat model belajar untuk menyesuaikan garis linear terbaik ke data, sehingga dapat memprediksi target (y) berdasarkan fitur (X).

4. **Membuat prediksi**:
   ```python
   predik = lr.predict(X_test)
   ```
   Setelah model regresi linear dilatih, kode ini menggunakan model yang telah dilatih (`lr`) untuk membuat prediksi pada data uji. `X_test` adalah matriks fitur dari data uji, dan hasil prediksi disimpan dalam variabel `predik`. Hasil prediksi ini adalah estimasi nilai target (y) berdasarkan fitur yang diberikan.

Dengan demikian, kode ini mengimplementasikan regresi linear, sebuah teknik pembelajaran mesin, yang digunakan untuk memodelkan hubungan linier antara fitur (variabel independen) dan target (variabel dependen) dalam data. Setelah melatih model menggunakan data latih, model tersebut dapat digunakan untuk membuat prediksi pada data yang belum pernah dilihat sebelumnya (data uji) untuk mengukur seberapa baik model ini memahami hubungan antara variabel-variabel tersebut.

``` bash
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
predik = lr.predict(X_test)
```

#### - EVALUASI DATA

Kode program di atas digunakan untuk mengukur akurasi atau performa model regresi linier pada data pengujian (test data) dan kemudian mencetak nilai akurasi tersebut ke layar. Berikut penjelasan komponen-komponen utama dari kode tersebut:

1. `nilai`: Ini adalah variabel yang digunakan untuk menyimpan hasil dari metode `score` yang dipanggil pada objek model regresi linier (`lr`). Biasanya, `score` mengembalikan nilai akurasi dari model, tetapi ini tergantung pada konteksnya dan mungkin dapat menjadi metrik lain seperti Mean Squared Error (MSE) atau R-squared (R^2), tergantung pada konfigurasi model dan tujuan analisisnya.

2. `lr.score(X_test, y_test)`: Ini adalah metode yang digunakan untuk menghitung akurasi model regresi linier pada data pengujian. Parameter yang dilewatkan ke metode ini adalah `X_test` dan `y_test`, yang mewakili fitur-fitur uji dan label target yang sesuai. Model akan melakukan prediksi berdasarkan fitur-fitur uji dan membandingkan hasil prediksi dengan label target yang sebenarnya. Metode `score` akan mengembalikan hasil pengukuran akurasi, biasanya dalam bentuk angka antara 0 hingga 1, di mana 1 menunjukkan akurasi sempurna.

3. `print('Akurasi Model Regresi Linier : ', nilai)`: Ini adalah pernyataan cetak yang digunakan untuk mencetak nilai akurasi model regresi linier ke layar. Pesan yang dicetak adalah "Akurasi Model Regresi Linier :" diikuti oleh nilai akurasi yang disimpan dalam variabel `nilai`.

Jadi, keseluruhan kode tersebut digunakan untuk mengukur dan menampilkan akurasi model regresi linier pada data pengujian, yang dapat membantu dalam mengevaluasi seberapa baik model ini dapat memprediksi nilai target berdasarkan fitur-fitur yang ada.

``` bash
nilai = lr.score(X_test, y_test)
print('Akurasi Model Regresi Linier : ', nilai)
```
Kode program di bawah merupakan contoh penggunaan modul `pickle` dalam Python untuk menyimpan (serialize) objek `lr` ke dalam file dengan nama `'estimasi_housing.sav'`. Mari kita jelaskan baris per baris:

1. `import pickle`: Ini adalah pernyataan impor yang memungkinkan Anda menggunakan modul `pickle` dalam kode Python. Modul `pickle` digunakan untuk menyimpan dan memuat objek Python ke atau dari file. Ini berguna ketika Anda ingin menyimpan objek yang kompleks, seperti model mesin pembelajaran atau struktur data yang rumit, ke dalam file untuk penggunaan nanti.

2. `filename = 'prediksi-harga-mobil.sav'`: Baris ini menetapkan nama file `'prediksi-harga-mobil.sav'` sebagai nama file yang akan digunakan untuk menyimpan objek.

3. `pickle.dump(lr, open(filename, 'wb'))`: Inilah inti dari kode. Dalam baris ini, dilakukan dua tindakan penting:
   - `pickle.dump(lr, ...)` digunakan untuk menyimpan objek `lr` ke dalam file.
   - `open(filename, 'wb')` digunakan untuk membuka file `'prediksi-harga-mobil.sav` dalam mode write binary ('wb'). Dengan menggunakan mode `'wb'`, Anda mengizinkan penulisan data biner ke file.

Jadi, kode ini akan mengambil objek yang disebut `lr` (mungkin adalah model regresi linear, sesuai dengan nama file `'prediksi-harga-mobil.sav'`) dan menyimpannya ke dalam file `'peprediksi-harga-mobil.sav'` dalam format biner. Ini adalah cara umum untuk menyimpan model mesin pembelajaran atau objek Python lainnya ke dalam file sehingga Anda dapat menggunakannya kembali di sesi Python berikutnya tanpa perlu melatih ulang model atau membangun ulang objek tersebut.

``` bash
import pickle
filename = 'prediksi-harga-mobil.sav'
pickle.dump(lr,open(filename,'wb'))
```
## Deployment

[Estimation APP](https://mesinlearning-5ajyhudcth6dbzmia4ymft.streamlit.app/).

