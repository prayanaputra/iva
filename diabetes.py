# %%
# Import Library yang dibutuhkan
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.metrics import classification_report_imbalanced

# %%
import pandas as pd

df = pd.read_csv("diabetes.csv")
df.head(5)

# %%
# Memeriksa missing values
print("DataFrame dengan missing values:")
print(df)

# Menghitung jumlah missing values di setiap kolom
missing_values_per_column = df.isnull().sum()
print("\nJumlah missing values di setiap kolom:")
print(missing_values_per_column)

# Menghitung jumlah total missing values dalam DataFrame
total_missing_values = df.isnull().sum().sum()
print("\nJumlah total missing values dalam DataFrame:")
print(total_missing_values)

# Menampilkan baris-baris yang memiliki missing values
rows_with_missing_values = df[df.isnull().any(axis=1)]
print("\nBaris dengan missing values:")
print(rows_with_missing_values)

# Menampilkan persentase missing values di setiap kolom
percentage_missing_values = (df.isnull().mean() * 100).round(2)
print("\nPersentase missing values di setiap kolom:")
print(percentage_missing_values)

# %%
# Memeriksa baris duplikat (mengembalikan boolean)
duplikat = df.duplicated()
print("\nBaris duplikat (boolean):")
print(duplikat)

# Menampilkan baris yang duplikat
baris_duplikat = df[df.duplicated()]
print("\nBaris yang duplikat:")
print(baris_duplikat)

# Menghitung jumlah baris duplikat
jumlah_duplikat = df.duplicated().sum()
print(f"\nJumlah total baris duplikat: {jumlah_duplikat}")

# Menghapus baris duplikat dan mempertahankan baris pertama yang muncul
df_dropped_duplicates = df.drop_duplicates()

print("\nDataFrame setelah menghapus baris duplikat (pertahankan yang pertama):")
print(df_dropped_duplicates)

# Menghapus baris duplikat dan mempertahankan baris terakhir yang muncul
df_dropped_duplicates_last = df.drop_duplicates(keep='last')

print("\nDataFrame setelah menghapus baris duplikat (pertahankan yang terakhir):")
print(df_dropped_duplicates_last)

# %%
# Memeriksa distribusi kelas
class_distribution = df['Outcome'].value_counts()
print("Distribusi kelas:")
print(class_distribution)

# Menampilkan persentase distribusi kelas
class_distribution_percentage = df['Outcome'].value_counts(normalize=True) * 100
print("\nPersentase distribusi kelas:")
print(class_distribution_percentage)

# %% [markdown]
# KNN PEMODELAN

# %%
# Menentukan Variabel X (Fitur/Atribut) dan Variabel y (Kelas/Label)

X= df.iloc[:, :-1]
y= df.values[:, -1]

pd.DataFrame(y).head()

# %%
from sklearn.model_selection import train_test_split

# Membagi data menjadi data training dan data testing
# Data untuk testing 25%, data untuk training 75%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# %%
# Inisiasi Model

model = KNeighborsClassifier(n_neighbors=7)

# Training model dengan .fit()

model.fit(X_train, y_train)

# %%
# Prediksi pada data test

y_pred = model.predict(X_test)
y_pred

# %%
y_test

# %%
# Memeriksa antara hasil prediksi dan data aktual

df = pd.DataFrame({'Prediksi': y_pred, 'Aktual': y_test})
df

# %%
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix

# %%
# Evaluasi Kinerja
print(f"accuracy_score {accuracy_score(y_test, y_pred)}")
print(classification_report_imbalanced(y_test, y_pred))

# %%
from sklearn.model_selection import cross_val_score

# Rentang nilai K yang akan diuji
k_range = range(1, 31)

# Menyimpan nilai rata-rata akurasi untuk setiap K
mean_scores = []

# Uji setiap nilai K menggunakan cross-validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    mean_scores.append(scores.mean())

# Mencari nilai K dengan akurasi tertinggi
optimal_k = k_range[np.argmax(mean_scores)]
optimal_accuracy = max(mean_scores)

print(f"Nilai K optimal untuk KNN adalah: {optimal_k}")
print(f"Akurasi tertinggi oleh nilai K optimal adalah: {optimal_accuracy:.4f}")

# Membuat grafik akurasi vs K
plt.figure(figsize=(10, 6))
plt.plot(k_range, mean_scores, marker='o')
plt.title('Akurasi vs Nilai K untuk KNN')
plt.xlabel('Nilai K')
plt.ylabel('Akurasi')
plt.xticks(np.arange(1, 31, 1))
plt.grid(True)
plt.show()


# %%
# Save the model to a file
import pickle # Import the pickle module

filename = 'diabetes_model.sav'
pickle.dump(model, open(filename, 'wb'))

print(f"Model telah disimpan ke dalam file {filename}")

# %%
import numpy as np
import seaborn as sns
import matplotlib
import sklearn
import imblearn
import pandas as pd

# Print version information
print(f"NumPy version: {np.__version__}")
print(f"Seaborn version: {sns.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Imbalanced-learn version: {imblearn.__version__}")
print(f"Pandas version: {pd.__version__}")



