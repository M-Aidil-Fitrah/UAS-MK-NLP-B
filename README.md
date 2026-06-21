# SkillScope ID

**Sistem Ekstraksi Skill Otomatis dari Lowongan Kerja Indonesia Menggunakan Named Entity Recognition (NER)**

SkillScope ID adalah proyek akhir mata kuliah **SINF6054 – Pemrosesan Bahasa Alami** yang membangun model NER berbasis transformer untuk mengekstrak **Hard Skill**, **Soft Skill**, dan **Technology** dari teks lowongan kerja berbahasa Indonesia. Model terbaik di-deploy langsung di browser melalui ONNX Runtime Web — tanpa memerlukan server backend.

---

## Daftar Isi

- [Fitur Utama](#fitur-utama)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Struktur Proyek](#struktur-proyek)
- [Dataset](#dataset)
- [Model](#model)
- [Hasil Evaluasi](#hasil-evaluasi)
- [Setup & Instalasi](#setup--instalasi)
  - [1. Prasyarat](#1-prasyarat)
  - [2. Clone Repository](#2-clone-repository)
  - [3. Download Model ONNX](#3-download-model-onnx)
  - [4. Menjalankan Web Demo](#4-menjalankan-web-demo)
  - [5. Menjalankan Notebook Training](#5-menjalankan-notebook-training)
- [Cara Penggunaan Web Demo](#cara-penggunaan-web-demo)
- [Contoh Input dan Output](#contoh-input-dan-output)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)

---

## Fitur Utama

- **Ekstraksi Skill Otomatis** — Mendeteksi dan mengkategorikan skill dari teks lowongan kerja ke dalam 3 kategori: Hard Skill, Soft Skill, dan Technology.
- **Client-Side Inference** — Seluruh proses inferensi berjalan langsung di browser menggunakan ONNX Runtime Web (WebAssembly). Data pengguna tidak pernah meninggalkan browser.
- **Highlight Entitas** — Teks lowongan di-highlight dengan warna berbeda per kategori (hijau untuk Hard Skill, biru untuk Technology, oranye untuk Soft Skill).
- **Confidence Score** — Setiap entitas yang terdeteksi dilengkapi dengan skor keyakinan model.
- **Export Hasil** — Hasil ekstraksi dapat di-export ke format JSON dan CSV.
- **Filter Kategori** — Pengguna dapat memfilter tampilan berdasarkan kategori entitas tertentu.
- **Contoh Siap Pakai** — Tersedia teks lowongan contoh untuk langsung dicoba.
- **Responsive Design** — Tampilan menyesuaikan layar desktop dan mobile dengan desain dark glassmorphism premium.

---

## Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (Client)                     │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │ Input    │──▸│ WordPiece    │──▸│ ONNX Runtime    │  │
│  │ Teks     │   │ Tokenizer    │   │ Web (WASM)      │  │
│  │ Lowongan │   │ (JavaScript) │   │                 │  │
│  └──────────┘   └──────────────┘   │  model.onnx     │  │
│                                    │ (NusaBERT-Large │  │
│                                    │  Quantized)     │  │
│  ┌──────────┐   ┌──────────────┐   └────────┬────────┘  │
│  │ Render   │◂──│ Post-process │◂───────────┘           │
│  │ Highlight│   │ BIO Merge &  │                        │
│  │ & Export │   │ Softmax      │                        │
│  └──────────┘   └──────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

Tidak ada server backend. Semua berjalan di sisi klien.

---

## Struktur Proyek

```
UAS/
├── README.md                          # Dokumentasi proyek (file ini)
├── .gitignore                         # File yang diabaikan Git
│
└── src/
    ├── notebooks/
    │   └── uasnlp.ipynb               # Notebook training & evaluasi (Kaggle)
    │
    ├── dataset/
    │   └── NERSkill.Id.txt            # Dataset NERSkill.Id (Download dari Mendeley Data)
    │
    └── web/
        ├── index.html                 # Halaman utama web demo
        ├── style.css                  # Styling (dark glassmorphism theme)
        ├── app.js                     # Logika aplikasi & inferensi ONNX
        ├── assets/
        │   └── favicon.png            # Ikon aplikasi
        └── model/
            ├── model.onnx             # Model NER ONNX quantized (~322 MB)
            ├── vocab.txt              # Kosakata tokenizer (~229 KB)
            ├── tokenizer.json         # Konfigurasi tokenizer (~984 KB)
            ├── tokenizer_config.json  # Pengaturan tokenizer (~261 KB)
            ├── config.json            # Konfigurasi arsitektur model
            ├── label_map.json         # Mapping ID ↔ Label NER
            └── special_tokens_map.json # Token spesial ([CLS], [SEP], dll)
```

> **Catatan:** File `model.onnx` (~322 MB) tidak di-track oleh Git karena ukurannya terlalu besar. File ini harus diunduh terpisah dari Google Drive (lihat bagian [Setup](#3-download-model-onnx)).

---

## Dataset

| Properti | Detail |
| --- | --- |
| Nama | **NERSkill.Id** |
| Sumber | Mendeley Data |
| DOI | `10.17632/5s8r9ndfvc.1` |
| Total Kalimat | 4.299 |
| Total Token | 418.868 |
| Total Entitas | 45.508 |
| Format | CoNLL / BIO tagging |
| Split | Train 80% · Validation 10% · Test 10% |

Label entitas yang digunakan:

| Label | Arti | Contoh |
| --- | --- | --- |
| `B-HSkill` / `I-HSkill` | Hard Skill | analisis data, pemrograman |
| `B-SSkill` / `I-SSkill` | Soft Skill | komunikasi, kepemimpinan |
| `B-Tech` / `I-Tech` | Technology | Python, MySQL, Laravel |
| `O` | Bukan entitas | Dibutuhkan, kandidat |

---

## Model

Dua model transformer Large (335M parameter) dibandingkan secara head-to-head:

| Model | HuggingFace ID | Deskripsi |
| --- | --- | --- |
| **IndoBERT-Large** | `indobenchmark/indobert-large-p2` | Model spesialis Bahasa Indonesia dari IndoNLU Benchmark |
| **NusaBERT-Large** | `LazarusNLP/NusaBERT-large` | Generasi penerus IndoBERT, dilatih dengan 16 miliar token |

Model pemenang (**NusaBERT-Large**) diekspor ke ONNX dan di-quantize:

| Format | Ukuran |
| --- | --- |
| ONNX FP32 | 1.280,78 MB |
| **ONNX INT8 Quantized** | **321,75 MB** (dipakai di web demo) |

---

## Hasil Evaluasi

### Perbandingan Model (Test Set)

| Metrik | IndoBERT-Large | NusaBERT-Large |
| --- | --- | --- |
| **F1 (micro avg)** | 0.7832 | **0.7904** 🏆 |
| Precision (micro) | 0.7699 | **0.7884** |
| Recall (micro) | **0.7969** | 0.7925 |
| Waktu Training | 4 jam 48 menit | **3 jam 22 menit** |

### Classification Report — NusaBERT-Large (Model Terpilih)

| Entity | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| HSkill | 0.7095 | 0.6949 | 0.7021 | 2.288 |
| SSkill | 0.7905 | 0.8602 | 0.8239 | 930 |
| Tech | 0.8584 | 0.8567 | 0.8575 | 2.498 |

---

## Setup & Instalasi

### 1. Prasyarat

- **Python 3.8+** (untuk menjalankan HTTP server lokal)
- **Browser modern** (Chrome, Firefox, atau Edge versi terbaru)
- Koneksi internet hanya dibutuhkan saat pertama kali memuat library ONNX Runtime Web dari CDN

### 2. Clone Repository

```bash
git clone https://github.com/M-Aidil-Fitrah/UAS-MK-NLP-B.git
cd UAS-MK-NLP-B
```

### 3. Download Model ONNX

File model ONNX (~322 MB) tidak disertakan dalam repository karena ukurannya yang besar. Download dari Google Drive:

**🔗 [Download Model ONNX dari Google Drive](https://drive.google.com/drive/folders/1KoJ85tszWcghGLPbV4i3U07S1foe4w3G?usp=sharing)**

Setelah di-download, letakkan file-file model ke dalam folder:

```
src/web/model/
```

Pastikan isi folder `model/` seperti berikut:

```
src/web/model/
├── model.onnx              # ← Download dari Google Drive
├── vocab.txt               # ✅ Sudah ada di repository
├── tokenizer.json          # ✅ Sudah ada di repository
├── tokenizer_config.json   # ✅ Sudah ada di repository
├── config.json             # ✅ Sudah ada di repository
├── label_map.json          # ✅ Sudah ada di repository
└── special_tokens_map.json # ✅ Sudah ada di repository
```

> **Penting:** Minimal file yang **wajib** ada agar web demo berjalan: `model.onnx`, `vocab.txt`, dan `label_map.json`.

### 4. Menjalankan Web Demo

Jalankan HTTP server lokal dari folder `src/web`:

```bash
cd src/web
python -m http.server 8080
```

Lalu buka browser dan akses:

```
http://localhost:8080
```

> ⚠️ **Jangan** membuka file `index.html` dengan double-click langsung. Browser akan memblokir `fetch()` pada protokol `file://`. Harus menggunakan HTTP server.

#### Alternatif Server

Jika tidak memiliki Python, gunakan alternatif lain:

```bash
# Menggunakan Node.js
npx serve src/web -p 8080

# Menggunakan PHP
php -S localhost:8080 -t src/web
```

### 5. Menjalankan Notebook Training

Notebook training dirancang untuk dijalankan di **Kaggle** dengan GPU T4:

1. Upload file `src/notebooks/uasnlp.ipynb` ke Kaggle.
2. Upload dataset `NERSkill.Id` sebagai Kaggle Dataset.
3. Aktifkan **GPU T4 × 2** sebagai accelerator.
4. Jalankan semua cell secara berurutan.
5. Setelah selesai (~8 jam), download folder `outputs/web_model/` dari output Kaggle.

Atau, install dependencies secara lokal (membutuhkan GPU):

```bash
pip install -r requirements.txt
```

---

## Cara Penggunaan Web Demo

1. **Buka** web demo di browser (`http://localhost:8080`).
2. **Tunggu** hingga model selesai dimuat (ditandai status "Model Ready ✓" berwarna hijau). Proses pertama kali memakan waktu ~20–60 detik.
3. **Masukkan teks** lowongan kerja Indonesia ke dalam textarea, atau klik tombol **"Muat Contoh"** untuk mengisi contoh otomatis.
4. **Klik "Extract Skills"** atau tekan **Ctrl + Enter**.
5. **Lihat hasil:**
   - Teks akan di-highlight dengan warna per kategori.
   - Entitas dikelompokkan ke dalam 3 kolom: Hard Skills, Technologies, Soft Skills.
   - Setiap entitas dilengkapi confidence score.
6. **Filter** hasil berdasarkan kategori menggunakan tombol filter (All / Hard / Tech / Soft).
7. **Export** hasil ke JSON atau CSV menggunakan tombol export di bagian bawah.

---

## Contoh Input dan Output

**Input:**

```text
PT Nusantara Data sedang mencari Data Analyst yang menguasai Python, SQL, dan
Microsoft Excel. Kandidat diharapkan memiliki kemampuan komunikasi yang baik,
teliti, mampu membuat dashboard, serta familiar dengan Tableau atau Power BI.
Pengalaman dalam analisis data dan machine learning menjadi nilai tambah.
Kemampuan bekerja dalam tim dan problem solving yang kuat sangat diutamakan.
```

**Output:**

| Kategori | Entitas | Confidence |
| --- | --- | --- |
| Hard Skill | analisis data | 94,5% |
| Hard Skill | machine learning | 96,1% |
| Technology | Python | 95,8% |
| Technology | Microsoft Excel | 97,1% |
| Technology | Tableau | 92,3% |
| Soft Skill | komunikasi | 96,8% |
| Soft Skill | tim | 93,7% |
| Soft Skill | problem solving | 70,8% |

---

## Teknologi yang Digunakan

### Training & Evaluasi

| Teknologi | Kegunaan |
| --- | --- |
| Python 3.10 | Bahasa pemrograman utama |
| PyTorch | Framework deep learning |
| HuggingFace Transformers | Fine-tuning model transformer |
| HuggingFace Datasets | Manajemen dataset |
| seqeval | Evaluasi metrik NER |
| ONNX / ONNX Runtime | Export & optimasi model |
| Optimum | Export transformer ke ONNX |
| Kaggle Notebook | Platform training (GPU T4 × 2) |

### Web Demo

| Teknologi | Kegunaan |
| --- | --- |
| HTML5 / CSS3 / JavaScript | Frontend web statis |
| ONNX Runtime Web | Inferensi model di browser (WebAssembly) |
| Google Fonts (Inter, JetBrains Mono) | Tipografi |

