# Time Series Forecasting with TensorFlow

Proyek ini mengeksplorasi prediksi deret waktu menggunakan TensorFlow, dengan dua pendekatan:

- ✅ **Univariate**: prediksi suhu menggunakan LSTM
- 🔧 **Multivariate**: prediksi dengan banyak fitur

---

## 📊 Dataset: Jena Climate Dataset
Dataset ini berasal dari stasiun cuaca Beutenberg di Jena, Jerman, dan tersedia di Kaggle melalui tautan berikut:

👉 [Jena Climate Dataset – Kaggle](https://www.kaggle.com/datasets/mnassrib/jena-climate)

## Univariate Model

**Arsitektur:**

```python
model = Sequential()
model.add(LSTM(60, input_shape=(60, 1), return_sequences=True))
model.add(LSTM(60))
model.add(Dense(30, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))

**Evaluai Model:**
- MAE : 0.500
- RMSE : 0.724
- R² : 0.992 ✅

## Multivariate Model

**Arsitektur:**

```python
model = Sequential()
model.add(Dense(64, input_shape=(N_PAST, N_FEATURES)))
model.add(Dense(32, activation='relu'))
model.add(Dense(N_FEATURES))

## 🚀 Cara Menjalankan

1. Clone repositori ini:
   ```bash
   git clone https://github.com/Ikram-sabila/Time-Series-Jena-Climate.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Jalankan notebook:
   ```bash
   jupyter notebook

4. Buka file notebook dan jalankan sel secara berurutan.
