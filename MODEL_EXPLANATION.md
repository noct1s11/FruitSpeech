# Penjelasan Pemilihan Model & Arsitektur (SVM & MLP)

Dokumen ini menjelaskan alasan teknis di balik pemilihan fitur audio Mel-Frequency Cepstral Coefficients (MFCC), serta penggunaan algoritma **Support Vector Machine (SVM)** dan **Multi-Layer Perceptron (MLP)** pada proyek FruitSpeech.

---

## 1. Mengapa Menggunakan MFCC + Machine Learning Tradisional?

### A. Mengapa Fitur MFCC?
**Mel-Frequency Cepstral Coefficients (MFCC)** adalah representasi spektral dari suara yang sangat populer dalam pemrosesan sinyal wicara (*speech processing*).
* **Representasi Pendengaran Manusia:** Skala Mel memetakan frekuensi audio agar sesuai dengan cara telinga manusia mendengar suara (lebih sensitif pada frekuensi rendah dibandingkan frekuensi tinggi).
* **Pereduksi Dimensi:** MFCC mengubah sinyal audio mentah (*raw waveform*) yang sangat padat menjadi representasi "sidik jari suara" (*voiceprint*) berdimensi rendah namun kaya akan informasi fonetik (karakter suku kata).
* **Independensi Pitch:** MFCC berfokus pada bentuk saluran suara (*vocal tract*) yang menghasilkan suara, bukan frekuensi dasar suara (*pitch*). Ini membantu model mendeteksi kata yang sama meskipun diucapkan oleh laki-laki (suara berat) atau perempuan (suara tinggi).

### B. Mengapa Tidak Menggunakan Deep Learning Skala Besar (seperti Transformers atau Wav2Vec)?
* **Ukuran Dataset Sangat Kecil (Data-Efficient):** Dataset saat ini hanya berkisar ~12-14 sampel per kelas buah (total ~126 sampel). Model Deep Learning berbasis Neural Network besar membutuhkan puluhan ribu sampel data agar tidak mengalami *overfitting* parah.
* **Efisiensi Komputasi:** Kombinasi MFCC dengan model ML tradisional (SVM/MLP) dapat dilatih dalam waktu kurang dari 1 detik di komputer lokal tanpa memerlukan GPU khusus, sementara akurasinya tetap tinggi untuk pengenalan kosakata terbatas (*isolated word recognition*).

---

## 2. Mengapa Memakai Support Vector Machine (SVM)?

**SVM** adalah salah satu algoritma klasifikasi terbaik untuk data berdimensi sedang dengan jumlah sampel yang sedikit.

* **Batas Keputusan yang Optimal (Maximum Margin Classifier):** SVM bekerja dengan mencari *hyperplane* pembatas yang memiliki jarak (margin) paling maksimum antar kelas buah. Pendekatan geometris ini membuat SVM sangat andal dan tidak mudah terpengaruh oleh *noise* (kebisingan latar belakang).
* **Tahan terhadap Overfitting:** Karakteristik matematis SVM membuatnya bekerja sangat baik meskipun jumlah fitur (52 dimensi) relatif besar dibandingkan jumlah data latih (126 sampel).
* **Keberhasilan Linear Kernel:** Pada proyek ini, **Linear SVM** memberikan akurasi tertinggi (**84.62%**). Ini menandakan bahwa fitur MFCC yang telah diperluas (mean + std + delta) sudah terpisah secara linear di ruang dimensi tinggi.

---

## 3. Mengapa Memakai Multi-Layer Perceptron (MLP)?

**MLP** adalah bentuk dasar dari Jaringan Saraf Tiruan (*Artificial Neural Network*) yang terdiri dari beberapa lapisan neuron tersembunyi (*hidden layers*).

* **Kemampuan Pemodelan Non-Linear:** Jika batas keputusan antar kelas suara sangat rumit dan tidak bisa dipisahkan dengan garis lurus (non-linear), MLP dapat mempelajari fungsi pemisah tersebut menggunakan fungsi aktivasi non-linear seperti ReLU.
* **Pembelajaran Representasi Hierarkis:** Melalui arsitektur bertingkat (misalnya, lapisan 128 neuron diikuti 64 neuron), MLP dapat mengombinasikan fitur-fitur MFCC dasar menjadi fitur tingkat tinggi yang lebih abstrak untuk membedakan pelafalan huruf vokal/konsonan.
* **Sebagai Pembanding/Benchmark:** Menyediakan MLP di samping SVM memberikan fleksibilitas bagi pengguna untuk membandingkan pendekatan berbasis geometri (SVM) dengan pendekatan berbasis jaringan saraf (MLP) untuk melihat mana yang berkinerja paling baik di lingkungan perekaman mereka.

---

## Ringkasan Perbandingan

| Fitur / Parameter | Support Vector Machine (SVM) | Multi-Layer Perceptron (MLP) |
| :--- | :--- | :--- |
| **Metode Kerja** | Pembatas Geometris Maksimum (*Hyperplane*) | Jaringan Saraf Tiruan dengan Gradien Turun |
| **Kelebihan Utama** | Sangat tangguh pada dataset kecil, hemat memori. | Mampu mempelajari hubungan non-linear yang sangat kompleks. |
| **Risiko** | Kurang optimal jika data sangat besar (>10.000). | Cenderung mudah *overfit* jika dataset terlalu sedikit tanpa regulasi. |
| **Akurasi Proyek** | **~84.62%** (Linear Kernel) | **~76.92%** |
