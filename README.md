# Klasifikasi Employee Attrition dengan AdaBoost

Repositori ini berisi kode program dan hasil evaluasi tugas individu Case Method mata kuliah Data Mining. Kasus yang dianalisis adalah klasifikasi risiko *employee attrition* menggunakan algoritma AdaBoost pada dataset IBM HR Analytics.

- **Nama:** Yusran Rizqi Laksono
- **NIM:** L0124125
- **Kelas:** A
- **Algoritma:** AdaBoost (Klasifikasi)

## Isi Repositori

- `adaboost_employee_attrition.py` —> skrip utama (EDA, preprocessing, Pipeline AdaBoost, 10-Fold Stratified Cross Validation, evaluasi test set, dan ekstraksi feature importance)
- `WA_Fn-UseC_-HR-Employee-Attrition.csv` —> dataset IBM HR Analytics (1.470 baris, 35 kolom)
- `evaluasi_adaboost.png` —> grafik Confusion Matrix dan Kurva ROC pada data uji (20%)
- `feature_importance_adaboost.png` —> grafik 15 fitur paling berpengaruh
- `requirements.txt` —> daftar pustaka Python yang digunakan

## Cara Menjalankan

Pastikan Python 3 sudah terpasang, lalu jalankan perintah berikut di direktori repositori:

```bash
pip install -r requirements.txt
python adaboost_employee_attrition.py
```

## Ringkasan Hasil

- **10-Fold Stratified Cross Validation:** Accuracy `0.8619 ± 0.0111`, Precision `0.8671 ± 0.1844`, Recall `0.1815 ± 0.0566`, F1-Score `0.2943 ± 0.0802`, ROC-AUC `0.8215 ± 0.0494`.
- **Test Set (20%):** Accuracy `0.8469`, Precision `0.5556`, Recall `0.2128`, F1-Score `0.3077`, ROC-AUC `0.7961`.
- **Faktor paling berpengaruh:** `OverTime_Yes` (0.1293), `TotalWorkingYears` (0.1021), `JobInvolvement` (0.0705), `MonthlyIncome` (0.0661), dan `NumCompaniesWorked` (0.0592).

## Sumber Dataset

IBM HR Analytics Employee Attrition & Performance (Kaggle): https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
