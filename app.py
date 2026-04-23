import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.title("Prediksi Hasil Panen Kelapa Sawit")

df = pd.read_csv("dataset_kelapa_sawit_500.csv")

rename_dict = {}

for col in df.columns:
    new_col = col.replace("_", " ").title()

    if "Suhu" in new_col:
        new_col += " (°C)"
    elif "Curah Hujan" in new_col:
        new_col += " (mm)"
    elif "%" in col or "Persen" in new_col:
        new_col += " (%)"

    new_col = new_col.replace("Per Ha", "Per Hektar")

    if "Hasil Panen" in new_col:
        new_col = "Hasil Panen (Ton/Hektar)"

    rename_dict[col] = new_col

df.rename(columns=rename_dict, inplace=True)

st.subheader("Dataset")
st.dataframe(df.head())

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Evaluasi Model")
st.write(f"MAE: {round(mae, 2)}")
st.write(f"R² Score: {round(r2, 2)}")

sns.set_style("whitegrid")

st.subheader("Visualisasi 1 - Distribusi Hasil Panen")
fig1, ax1 = plt.subplots()
sns.histplot(y, kde=True, ax=ax1)
ax1.set_title("Distribusi Hasil Panen Sawit")
ax1.set_xlabel(y.name)
ax1.set_ylabel("Frekuensi")
st.pyplot(fig1)

st.subheader("Visualisasi 2 - Korelasi Antar Variabel")
fig2, ax2 = plt.subplots(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax2)
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)
st.pyplot(fig2)

st.subheader("Visualisasi 3 - Perbandingan Aktual vs Prediksi")
fig3, ax3 = plt.subplots()
ax3.scatter(y_test, y_pred)

ax3.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)

ax3.set_xlabel("Nilai Aktual")
ax3.set_ylabel("Nilai Prediksi")
ax3.set_title("Perbandingan Aktual vs Prediksi")
st.pyplot(fig3)

st.subheader("Simulasi Hasil Panen")

input_data = {}

for col in X.columns:
    input_data[col] = st.number_input(
        f"{col}",
        value=float(X[col].mean())
    )

input_df = pd.DataFrame([input_data])

if st.button("Hitung Prediksi"):
    hasil = model.predict(input_df)[0]

    if hasil < y.quantile(0.33):
        kategori = "Rendah"
        st.error(f"Kategori Produksi: {kategori}")
    elif hasil < y.quantile(0.66):
        kategori = "Sedang"
        st.warning(f"Kategori Produksi: {kategori}")
    else:
        kategori = "Tinggi"
        st.success(f"Kategori Produksi: {kategori}")

    st.success(f"Estimasi Hasil Panen: {round(hasil, 2)} Ton/Hektar")
