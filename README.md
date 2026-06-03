# 🍎 FruitSpeech - Aplikasi ASR & TTS Interaktif Bahasa Indonesia

FruitSpeech adalah aplikasi konversi suara dan teks dua arah (dua arah) berbasis web yang dirancang khusus untuk mendeteksi suara buah-buahan dalam Bahasa Indonesia (ASR) serta menyintesis teks menjadi ucapan suara alami (TTS) dengan kontrol pemrosesan sinyal digital (DSP) secara real-time.

Aplikasi ini dibangun menggunakan antarmuka grafis (GUI) modern berbasis **Streamlit** dengan pendekatan desain **Fruity UI** yang cerah, responsif, dan interaktif.

---

## 🚀 Fitur Utama

### 1. Modul ASR (Automatic Speech Recognition)
Mengubah ucapan/suara nama buah berbahasa Indonesia dari mikrofon secara otomatis menjadi teks.
* **Ekstraksi Fitur**: Menggunakan MFCC (Mel-Frequency Cepstral Coefficients) sebanyak 13 koefisien menggunakan library `librosa`.
* **Klasifikasi Mandiri**: Tanpa library ASR siap pakai (black-box), model dilatih secara mandiri menggunakan pilihan arsitektur **SVM (Support Vector Machine)** atau **MLP (Multi-Layer Perceptron / Neural Network)** melalui `scikit-learn`.
* **Cakupan Terbatas**: Mengenali **10 kelas kata buah** Bahasa Indonesia: *apel, jeruk, mangga, pisang, semangka, melon, anggur, pepaya, nanas, dan stroberi*.
* **Real-time Inference**: Merekam suara langsung dari mikrofon selama 2 detik lalu mendeteksi kata yang diucapkan.

### 2. Modul TTS (Text-to-Speech)
Mengubah input teks bebas Bahasa Indonesia menjadi ucapan suara yang alami.
* **Bahasa Indonesia**: Output vokal dikonversi secara alami dalam logat Indonesia (berbasis engine `gTTS`).
* **Pengaturan Kecepatan**: Pilihan kecepatan bicara lambat (*time-stretch* 0.75x), normal (1.0x), atau cepat (*time-stretch* 1.30x) secara presisi menggunakan pemrosesan DSP.
* **Ekspor Audio**: Hasil suara dapat diputar langsung di browser dan diunduh ke dalam format berkas `.wav` berkualitas tinggi.

### 3. Fitur Tambahan (Menerapkan 4 Fitur Tambahan)
* **Pilihan Gender Suara (Laki-laki / Perempuan)**: Merekayasa pitch (tinggi nada) suara vokal secara lokal menggunakan algoritma *pitch-shifting* dari `librosa` untuk menghasilkan karakter suara laki-laki (pitch dalam) dan perempuan (pitch tinggi).
* **Confidence Score (Skor Keyakinan)**: Menampilkan tingkat keyakinan prediksi model ASR dalam bentuk persentase metrik dan visualisasi diagram batang probabilitas untuk semua kelas buah.
* **Visualisasi MFCC**: Menampilkan grafik visualisasi berupa **Heatmap Spektrogram MFCC (2D)** dari audio perekaman serta **Grafik Garis Rata-rata MFCC (1D)** sebagai representasi sidik jari suara (*voiceprint*).
* **Integrasi ASR ➡️ TTS**: Hasil deteksi suara dari modul ASR dapat disuarakan kembali menggunakan modul TTS secara otomatis atau disalin langsung ke kolom teks editor TTS dengan satu klik.

---

## 📁 Struktur Direktori Proyek

```directory
FruitSpeech/
│
├── .streamlit/
│   └── config.toml          # Konfigurasi tema global Streamlit (Fruity light mode)
│
├── dataset/                 # Folder penyimpanan rekaman sampel audio (.wav) latih
│   ├── apel/
│   ├── jeruk/
│   └── ... (kelas buah lainnya)
│
├── models/
│   └── fruit_classifier.pkl # File model klasifikasi yang telah dilatih & metadata
│
├── utils/
│   ├── audio_recorder.py    # Utilitas perekaman audio lokal
│   ├── feature_extraction.py# Ekstraksi koefisien MFCC dari audio
│   └── tts.py               # Pemrosesan DSP TTS (pitch-shifting & time-stretching)
│
├── app.py                   # Entry point aplikasi Web GUI (Streamlit)
├── app_cli.py               # Entry point aplikasi CLI terminal (versi cadangan lama)
├── train_model.py           # Pipeline latih model klasifikasi (SVM/MLP)
├── predict.py               # Pipeline inferensi prediksi kelas audio
└── requirements.txt         # Daftar pustaka dependensi Python
```

---

## 🔧 Instalasi dan Persiapan

### Kebutuhan Sistem
* Python 3.8 s.d 3.11
* Mikrofon internal/eksternal yang aktif

### Langkah Instalasi

1. **Clone repositori ini**:
   ```bash
   git clone <link-repo-github>
   cd FruitSpeech
   ```

2. **Buat & aktifkan virtual environment**:
   * Di macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   * Di Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Cara Menjalankan Aplikasi

### Menjalankan Web GUI (Streamlit)
Jalankan perintah berikut di terminal:
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka pada browser Anda di alamat:
**[http://localhost:8501](http://localhost:8501)**

### Menjalankan Versi CLI Terminal (Opsional)
Jika ingin menjalankan program dalam mode command-line:
```bash
python app_cli.py
```

---

## 📖 Panduan Penggunaan Aplikasi di GUI

### Langkah 1: Merekam Dataset Latih
1. Navigasi ke halaman **⚙️ MANAJEMEN DATASET & MODEL** melalui menu navigasi di bagian atas.
2. Pada bagian **Rekam Sampel Latih Baru**, pilih buah yang ingin Anda rekam (misalnya: *apel*).
3. Klik tombol **🎙️ Rekam Sampel**. Anda akan mendengar instruksi suara menyebutkan kata buah.
4. Ucapkan kata buah tersebut ke arah mikrofon. Rekaman akan berjalan selama 2 detik dengan visual indikator progress bar.
5. Lakukan perekaman minimal 3-5 sampel untuk setiap nama buah agar dataset memiliki variabilitas data.

### Langkah 2: Melatih Model Klasifikasi
1. Setelah sampel audio terkumpul, di halaman **⚙️ MANAJEMEN DATASET & MODEL**, pilih jenis model klasifikasi: **SVM** atau **MLP**.
2. Klik tombol **⚡ Latih Model Sekarang**.
3. Sistem akan mengekstrak fitur MFCC dari seluruh file audio WAV di folder `dataset/` dan melatih model.
4. Nilai akurasi pengujian, laporan evaluasi presisi per kelas, dan visualisasi **Confusion Matrix Heatmap** akan ditampilkan di panel kanan.

### Langkah 3: Melakukan Deteksi Suara (ASR)
1. Pindah ke halaman **🎙️ DETEKSI SUARA (ASR)** melalui navigasi atas.
2. Pilih metode input: **Rekam Langsung (Mikrofon)** atau **Unggah File Audio (.wav)**.
3. Jika merekam langsung, klik **🎙️ Mulai Merekam Suara (2 Detik)** dan sebutkan nama buah yang telah dilatih (misalnya: "pisang").
4. Hasil prediksi buah akan tampil dalam teks gradasi neon berukuran besar.
5. Nilai **Confidence Score** dan grafik Spektrogram MFCC suara Anda akan muncul secara detail.
6. Anda dapat menekan **Bacakan Hasil Prediksi (TTS)** untuk memicu asisten menyuarakan hasil klasifikasi.

### Langkah 4: Sintesis Suara (TTS)
1. Buka halaman **🗣️ SINTESIS SUARA (TTS)**.
2. Ketik kalimat apa saja dalam Bahasa Indonesia pada kolom yang disediakan.
3. Atur preferensi suara: **Gender** (Laki-laki / Perempuan) dan **Kecepatan** (Lambat / Normal / Cepat).
4. Klik **🗣️ Sintesis & Putar Suara** untuk mendengarkan pelafalan vokal secara langsung dari browser.
5. Klik **💾 Unduh Hasil Suara (.wav)** jika ingin menyimpan berkas audio tersebut secara lokal.
