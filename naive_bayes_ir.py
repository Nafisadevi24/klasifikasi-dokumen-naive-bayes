# =====================================================
# INFORMATION RETRIEVAL - NAIVE BAYES
# KLASIFIKASI DAN PENCARIAN DOKUMEN TEKS
# =====================================================

# =====================================================
# IMPORT LIBRARY
# =====================================================

import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# LIBRARY STEMMING BAHASA INDONESIA
# =====================================================

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =====================================================
# MEMBUAT STEMMER
# =====================================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# =====================================================
# STOPWORD
# =====================================================

stopwords = [
    "dan", "di", "ke", "dari", "yang",
    "untuk", "dengan", "atau", "pada",
    "adalah", "ini", "itu", "dalam",
    "secara", "oleh", "karena", "agar",
    "sebagai", "bahwa", "juga", "lebih",
    "dapat", "menjadi", "antara", "setiap",
    "suatu", "akan", "telah", "sudah",
    "para", "mereka", "kami", "kita",
    "anda", "ia", "saya", "aku"
]

# =====================================================
# DATASET PATH
# =====================================================

dataset_path = "."

# =====================================================
# CEK DATASET
# =====================================================

if not os.path.exists(dataset_path):

    print("Folder DATA_DOKUMEN tidak ditemukan!")
    exit()

# =====================================================
# LIST DATA
# =====================================================

documents = []
labels = []
file_names = []

# =====================================================
# MEMBACA DATASET
# =====================================================

print("\n================================================")
print("MEMBACA DATASET DOKUMEN")
print("================================================")

for category in os.listdir(dataset_path):

    category_path = os.path.join(dataset_path, category)

    if os.path.isdir(category_path):

        print(f"\nKategori : {category}")

        for filename in os.listdir(category_path):

            file_path = os.path.join(category_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:

                text = file.read()

                # =================================================
                # CEK FILE KOSONG
                # =================================================

                if text.strip() == "":
                    print(f"\nFile {filename} kosong, dilewati.")
                    continue

                print(f"\nFile : {filename}")
                print(f"Isi Dokumen :\n{text}")

                documents.append(text)
                labels.append(category)
                file_names.append(filename)

# =====================================================
# MENAMPILKAN DATASET
# =====================================================

print("\n================================================")
print("DATASET")
print("================================================")

df = pd.DataFrame({
    "File": file_names,
    "Dokumen": documents,
    "Kategori": labels
})

print(df)

# =====================================================
# INFORMASI DATASET
# =====================================================

print("\n================================================")
print("INFORMASI DATASET")
print("================================================")

print(f"Total Dokumen : {len(documents)}")

kategori_unik = sorted(set(labels))

for kelas in kategori_unik:

    jumlah = labels.count(kelas)

    print(f"Kategori {kelas} : {jumlah} dokumen")

print("\nKategori Dataset :")
print(kategori_unik)

# =====================================================
# PANJANG DOKUMEN
# =====================================================

print("\n================================================")
print("PANJANG DOKUMEN")
print("================================================")

for i in range(len(documents)):

    jumlah_kata = len(documents[i].split())

    print(f"{file_names[i]} = {jumlah_kata} kata")

# =====================================================
# TEXT PREPROCESSING
# =====================================================

print("\n================================================")
print("TEXT PREPROCESSING")
print("================================================")

preprocessed_documents = []

for text in documents:

    print("\n------------------------------------------------")
    print("DOKUMEN ASLI :")
    print(text)

    # =================================================
    # CASE FOLDING
    # =================================================

    case_folding = text.lower()

    print("\nCase Folding :")
    print(case_folding)

    # =================================================
    # REMOVE SYMBOL / ANGKA
    # =================================================

    remove_symbol = re.sub(r'[^a-zA-Z\s]', '', case_folding)

    print("\nRemove Symbol :")
    print(remove_symbol)

    # =================================================
    # TOKENIZING
    # =================================================

    tokenizing = remove_symbol.split()

    print("\nTokenizing :")
    print(tokenizing)

    print("\nJumlah Token :")
    print(len(tokenizing))

    # =================================================
    # STOPWORD REMOVAL
    # =================================================

    filtered_words = []

    for word in tokenizing:

        if word not in stopwords:
            filtered_words.append(word)

    print("\nStopword Removal :")
    print(filtered_words)

    print("\nJumlah Setelah Stopword Removal :")
    print(len(filtered_words))

    # =================================================
    # STEMMING
    # =================================================

    stemming_result = []

    for word in filtered_words:

        stem_word = stemmer.stem(word)

        stemming_result.append(stem_word)

    print("\nStemming :")
    print(stemming_result)

    # =================================================
    # FINAL TEXT
    # =================================================

    final_text = " ".join(stemming_result)

    print("\nFinal Text :")
    print(final_text)

    preprocessed_documents.append(final_text)

# =====================================================
# HASIL PREPROCESSING
# =====================================================

print("\n================================================")
print("HASIL AKHIR PREPROCESSING")
print("================================================")

for i in range(len(preprocessed_documents)):

    print(f"\nFile : {file_names[i]}")
    print(preprocessed_documents[i])

# =====================================================
# TF-IDF VECTORIZER
# =====================================================

print("\n================================================")
print("TF-IDF VECTORIZER")
print("================================================")

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(preprocessed_documents)

fitur = vectorizer.get_feature_names_out()

# =====================================================
# VOCABULARY SIZE
# =====================================================

print(f"\nVocabulary Size : {len(fitur)}")

print("\nJumlah Fitur :", len(fitur))

print("\nDaftar Fitur / Kata :")
print(fitur)

# =====================================================
# UKURAN MATRIX TF-IDF
# =====================================================

print("\n================================================")
print("UKURAN MATRIX TF-IDF")
print("================================================")

print(f"Jumlah Baris (Dokumen) : {X.shape[0]}")
print(f"Jumlah Kolom (Fitur)   : {X.shape[1]}")

# =====================================================
# PENJELASAN TF-IDF
# =====================================================

print("\n================================================")
print("PENJELASAN TF-IDF")
print("================================================")

print("""
TF-IDF digunakan untuk mengubah dokumen teks
menjadi data numerik.

TF (Term Frequency):
Menghitung jumlah kemunculan kata
pada dokumen.

IDF (Inverse Document Frequency):
Menghitung tingkat kepentingan kata
berdasarkan seluruh dokumen.

Kata yang sering muncul
pada dokumen tertentu
akan memiliki bobot besar.

Kata umum yang muncul
pada semua dokumen
akan memiliki bobot kecil.

TF-IDF menghasilkan representasi teks
yang lebih baik dibanding
frekuensi biasa.
""")

# =====================================================
# HASIL TF-IDF
# =====================================================

print("\n================================================")
print("HASIL TF-IDF")
print("================================================")

tfidf_df = pd.DataFrame(
    X.toarray(),
    columns=fitur
)

print(tfidf_df)

# =====================================================
# SPLIT DATA TRAINING DAN TESTING
# =====================================================

print("\n================================================")
print("SPLIT DATA TRAINING DAN TESTING")
print("================================================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

print(f"Jumlah Data Training : {X_train.shape[0]}")
print(f"Jumlah Data Testing  : {X_test.shape[0]}")

# =====================================================
# TRAINING MODEL NAIVE BAYES
# =====================================================

print("\n================================================")
print("TRAINING MODEL NAIVE BAYES")
print("================================================")

model = MultinomialNB()

model.fit(X_train, y_train)

print("Training model berhasil dilakukan!")

# =====================================================
# INPUT QUERY
# =====================================================

print("\n================================================")
print("MESIN PENCARIAN DOKUMEN")
print("================================================")

query = input("Masukkan Query Pencarian : ")

print("\nQuery :")
print(query)

# =====================================================
# PREPROCESSING QUERY
# =====================================================

print("\n================================================")
print("PREPROCESSING QUERY")
print("================================================")

# CASE FOLDING

test = query.lower()

print("\nCase Folding :")
print(test)

# REMOVE SYMBOL

test = re.sub(r'[^a-zA-Z\s]', '', test)

print("\nRemove Symbol :")
print(test)

# TOKENIZING

tokens = test.split()

print("\nTokenizing :")
print(tokens)

# =====================================================
# STOPWORD REMOVAL
# =====================================================

filtered = []

for word in tokens:

    if word not in stopwords:
        filtered.append(word)

print("\nStopword Removal :")
print(filtered)

# =====================================================
# STEMMING
# =====================================================

stemmed = []

for word in filtered:

    stemmed_word = stemmer.stem(word)

    stemmed.append(stemmed_word)

print("\nStemming :")
print(stemmed)

# =====================================================
# FINAL QUERY
# =====================================================

final_test = " ".join(stemmed)

print("\nFinal Query :")
print(final_test)

# =====================================================
# TRANSFORM QUERY TF-IDF
# =====================================================

print("\n================================================")
print("TRANSFORM QUERY TF-IDF")
print("================================================")

new_X = vectorizer.transform([final_test])

print("Transform query berhasil!")

# =====================================================
# PREDIKSI KATEGORI
# =====================================================

prediction = model.predict(new_X)

probability = model.predict_proba(new_X)

# =====================================================
# HASIL PROBABILITAS
# =====================================================

print("\n================================================")
print("HASIL PROBABILITAS")
print("================================================")

for i in range(len(model.classes_)):

    persen = probability[0][i] * 100

    print(f"{model.classes_[i]} = {persen:.2f}%")

# =====================================================
# HASIL KLASIFIKASI
# =====================================================

print("\n================================================")
print("HASIL KLASIFIKASI")
print("================================================")

print(f"Query             : {query}")
print(f"Kategori Prediksi : {prediction[0]}")

# =====================================================
# COSINE SIMILARITY
# =====================================================

print("\n================================================")
print("PERHITUNGAN COSINE SIMILARITY")
print("================================================")

similarity = cosine_similarity(new_X, X)

print(similarity)

# =====================================================
# DOKUMEN PALING RELEVAN
# =====================================================

print("\n================================================")
print("DOKUMEN PALING RELEVAN")
print("================================================")

hasil = []

for i in range(len(documents)):

    hasil.append([
        file_names[i],
        labels[i],
        similarity[0][i],
        documents[i]
    ])

hasil = sorted(hasil, key=lambda x: x[2], reverse=True)

ranking = 1
top_k = 5

for item in hasil[:top_k]:

    if item[2] > 0:

        persen_similarity = item[2] * 100

        print(f"\nRank : {ranking}")
        print(f"File : {item[0]}")
        print(f"Kategori : {item[1]}")
        print(f"Similarity Score : {persen_similarity:.2f}%")
        print(f"Isi Dokumen :\n{item[3]}")

        ranking += 1

if ranking == 1:

    print("Dokumen relevan tidak ditemukan.")

# =====================================================
# EVALUASI MODEL
# =====================================================

print("\n================================================")
print("EVALUASI MODEL")
print("================================================")

train_pred = model.predict(X_train)

y_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_pred)

test_accuracy = accuracy_score(y_test, y_pred)

print(f"Training Accuracy : {train_accuracy:.4f}")
print(f"Testing Accuracy  : {test_accuracy:.4f}")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\n================================================")
print("CLASSIFICATION REPORT")
print("================================================")

print(classification_report(y_test, y_pred))

# =====================================================
# CONFUSION MATRIX
# =====================================================

print("\n================================================")
print("CONFUSION MATRIX")
print("================================================")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=model.classes_,
    columns=model.classes_
)

print(cm_df)

# =====================================================
# VISUALISASI CONFUSION MATRIX
# =====================================================

plt.figure(figsize=(6,5))

sns.heatmap(
    cm_df,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix Naive Bayes")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()