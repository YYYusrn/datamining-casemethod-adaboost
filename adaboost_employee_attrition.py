"""
Tugas Individu Case Method - Data Mining
Nama      : Yusran Rizqi Laksono
NIM       : L0124125
Kelas     : A
Algoritma : AdaBoost (Klasifikasi)
Dataset   : IBM HR Analytics Employee Attrition & Performance
Sumber    : https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Petunjuk penggunaan:
1. Unduh file 'WA_Fn-UseC_-HR-Employee-Attrition.csv' dari Kaggle.
2. Letakkan file CSV tersebut di folder yang sama dengan skrip ini.
3. Pastikan library terpasang: pip install pandas numpy scikit-learn matplotlib seaborn
4. Jalankan: python adaboost_employee_attrition.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


#1. Memuat Dataset dan Eksplorasi Awal (EDA)
DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"File '{DATA_PATH}' tidak ditemukan. "
        "Unduh dari Kaggle (IBM HR Analytics Employee Attrition) "
        "dan letakkan di direktori yang sama dengan skrip ini."
    )

df = pd.read_csv(DATA_PATH)

print("=== 1. EKSPLORASI AWAL DATASET ===")
print(f"Ukuran dataset       : {df.shape[0]} baris x {df.shape[1]} kolom")
print(f"Total missing values : {df.isnull().sum().sum()}")
print(f"Total baris duplikat : {df.duplicated().sum()}")
print("\nDistribusi kelas target (Attrition):")
target_counts = df["Attrition"].value_counts()
target_pct = df["Attrition"].value_counts(normalize=True) * 100
for label in target_counts.index:
    print(f"  - {label:<3}: {target_counts[label]} karyawan ({target_pct[label]:.2f}%)")

#2. Data Preprocessing
kolom_hapus = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
df_clean = df.drop(columns=kolom_hapus, errors="ignore").copy()

#encoding target biner: Yes -> 1 (keluar/attrition), No -> 0 (tidak keluar)
y = df_clean["Attrition"].map({"Yes": 1, "No": 0})
X = df_clean.drop(columns=["Attrition"])

#identifikasi kolom kategorikal dan numerikal
fitur_numerikal = X.select_dtypes(include=["number"]).columns.tolist()
fitur_kategorikal = X.select_dtypes(exclude=["number"]).columns.tolist()

print("\n=== 2. DATA PREPROCESSING ===")
print(f"Jumlah fitur numerikal   : {len(fitur_numerikal)}")
print(f"Jumlah fitur kategorikal : {len(fitur_kategorikal)}")

#ColumnTransformer agar encoding dilakukan aman di dalam Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", fitur_numerikal),
        (
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            fitur_kategorikal,
        ),
    ]
)

#model AdaBoost dengan Decision Stump (max_depth=1) sebagai weak learner
adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", adaboost_model),
    ]
)

#3. Validasi Model: 10-Fold Stratified Cross Validation
print("\n=== 3. HASIL 10-FOLD STRATIFIED CROSS VALIDATION ===")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

cv_results = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
)

df_cv = pd.DataFrame(
    {
        "Fold": range(1, 11),
        "Accuracy": cv_results["test_accuracy"],
        "Precision": cv_results["test_precision"],
        "Recall": cv_results["test_recall"],
        "F1-Score": cv_results["test_f1"],
        "ROC-AUC": cv_results["test_roc_auc"],
    }
)
print(df_cv.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\nRata-rata 10-Fold Cross Validation (± Standar Deviasi):")
for metrik in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
    mean_val = df_cv[metrik].mean()
    std_val = df_cv[metrik].std()
    print(f"  - {metrik:<10}: {mean_val:.4f} ± {std_val:.4f}")

#4. Evaluasi Detail pada Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n=== 4. EVALUASI DETAIL (TEST SET 20%) ===")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Bertahan (0)", "Attrition (1)"]))

#simpan grafik Confusion Matrix & Kurva ROC
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Prediksi Bertahan", "Prediksi Attrition"],
    yticklabels=["Aktual Bertahan", "Aktual Attrition"],
    ax=axes[0],
)
axes[0].set_title("Confusion Matrix - AdaBoost")
axes[0].set_xlabel("Label Prediksi")
axes[0].set_ylabel("Label Aktual")

axes[1].plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AdaBoost (AUC = {auc_score:.4f})")
axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Baseline Acak")
axes[1].set_title("Kurva ROC - AdaBoost")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(loc="lower right")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("evaluasi_adaboost.png", dpi=150)
plt.close()


#5. Analisis Faktor Paling Berpengaruh (Feature Importance)
ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
nama_fitur_kat = ohe.get_feature_names_out(fitur_kategorikal).tolist()
semua_nama_fitur = fitur_numerikal + nama_fitur_kat

importances = pipeline.named_steps["classifier"].feature_importances_
df_importance = (
    pd.DataFrame({"Fitur": semua_nama_fitur, "Importance": importances})
    .sort_values(by="Importance", ascending=False)
    .reset_index(drop=True)
)

top_15 = df_importance.head(15)
print("\n=== 5. TOP 15 FAKTOR PALING BERPENGARUH (FEATURE IMPORTANCE) ===")
print(top_15.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

#visualisasi top 15 feature importance
plt.figure(figsize=(9, 6))
sns.barplot(data=top_15, x="Importance", y="Fitur", hue="Fitur", palette="viridis", legend=False)
plt.title("Top 15 Faktor Paling Berpengaruh terhadap Employee Attrition (AdaBoost)")
plt.xlabel("Skor Feature Importance")
plt.ylabel("Fitur")
plt.tight_layout()
plt.savefig("feature_importance_adaboost.png", dpi=150)
plt.close()


#6. Identifikasi Karyawan Berisiko Tinggi untuk Rekomendasi Manajemen
hasil_risiko = X_test.copy()
hasil_risiko["Aktual_Attrition"] = y_test.values
hasil_risiko["Probabilitas_Attrition"] = y_proba
hasil_risiko["Prediksi_Attrition"] = y_pred

kolom_ringkas = [
    col
    for col in ["Age", "Department", "JobRole", "MonthlyIncome", "OverTime", "YearsAtCompany"]
    if col in hasil_risiko.columns
] + ["Probabilitas_Attrition", "Aktual_Attrition"]

top_risiko = hasil_risiko.sort_values(by="Probabilitas_Attrition", ascending=False).head(5)
print("\n=== 6. TOP 5 KARYAWAN BERISIKO TINGGI (TEST SET) ===")
print(top_risiko[kolom_ringkas].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\nSelesai! Grafik tersimpan sebagai:")
print("  - evaluasi_adaboost.png")
print("  - feature_importance_adaboost.png")
