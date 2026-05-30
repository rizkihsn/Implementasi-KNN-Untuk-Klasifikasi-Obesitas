import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# 1. LOAD DATASET
# ============================================================
df = pd.read_csv('Data/ObesityDataSet_raw_and_data_sinthetic.csv')
print(f"Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

# ============================================================
# 2. PREPROCESSING — Encode Categorical Columns
# ============================================================
label_encoders = {}
categorical_cols = [
    'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',
    'family_history_with_overweight', 'CAEC', 'MTRANS', 'NObeyesdad'
]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print("Encoding kategorikal selesai.")

# ============================================================
# 3. SPLIT DATASET — Train (70%), Validasi (15%), Test (15%)
# ============================================================
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Step 1: Pisahkan train (70%) dan sementara (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
# Step 2: Pisahkan sementara menjadi validasi (15%) dan test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nPembagian Dataset:")
print(f"  Set Pelatihan (Train) : {X_train.shape[0]} sampel ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Set Validasi (Val)    : {X_val.shape[0]} sampel ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"  Set Pengujian (Test)  : {X_test.shape[0]} sampel ({X_test.shape[0]/len(X)*100:.1f}%)")

# ============================================================
# 4. PREPROCESSING — Normalisasi (StandardScaler)
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Fit HANYA di train
X_val_scaled   = scaler.transform(X_val)          # Transform val
X_test_scaled  = scaler.transform(X_test)         # Transform test
print("\nNormalisasi (StandardScaler) selesai.")

# ============================================================
# 5. PREPROCESSING — Feature Selection (SelectKBest)
# ============================================================
# Gunakan f_classif (ANOVA F-test) untuk memilih 12 fitur terbaik dari 16
selector = SelectKBest(score_func=f_classif, k=12)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)  # Fit HANYA di train
X_val_selected   = selector.transform(X_val_scaled)
X_test_selected  = selector.transform(X_test_scaled)

# Tampilkan fitur yang dipilih
feature_names = X.columns.tolist()
selected_mask = selector.get_support()
selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
removed_features  = [feature_names[i] for i in range(len(feature_names)) if not selected_mask[i]]

print(f"\nFeature Selection (SelectKBest, k=12):")
print(f"  Fitur yang DIPILIH ({len(selected_features)}) : {selected_features}")
print(f"  Fitur yang DIBUANG ({len(removed_features)})  : {removed_features}")

# ============================================================
# 6. PARAMETER TUNING — GridSearchCV (5-fold CV pada Train Set)
# ============================================================
print("\nMemulai Parameter Tuning (GridSearchCV) dengan 5-fold Cross Validation...")
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights'    : ['uniform', 'distance'],
    'metric'     : ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_selected, y_train)

best_knn = grid_search.best_estimator_
print(f"\nParameter KNN Terbaik : {grid_search.best_params_}")
print(f"CV Score (Validasi 5-fold) : {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")

# ============================================================
# 7. EVALUASI MODEL
# ============================================================
# Evaluasi pada Set VALIDASI
y_val_pred  = best_knn.predict(X_val_selected)
val_accuracy = accuracy_score(y_val, y_val_pred)

# Evaluasi pada Set PENGUJIAN (Test)
y_test_pred  = best_knn.predict(X_test_selected)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\n{'='*50}")
print(f"HASIL EVALUASI MODEL")
print(f"{'='*50}")
print(f"  Akurasi Set Validasi (Val)   : {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"  Akurasi Set Pengujian (Test) : {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"{'='*50}")

print("\nLaporan Klasifikasi (Test Set):\n")
print(classification_report(
    y_test, y_test_pred,
    target_names=label_encoders['NObeyesdad'].classes_
))

# ============================================================
# 8. SIMPAN METRIK
# ============================================================
metrics = {
    "accuracy"        : test_accuracy,
    "val_accuracy"    : val_accuracy,
    "cv_score"        : grid_search.best_score_,
    "best_params"     : grid_search.best_params_,
    "selected_features": selected_features,
    "removed_features" : removed_features,
    "train_size"      : X_train.shape[0],
    "val_size"        : X_val.shape[0],
    "test_size"       : X_test.shape[0]
}
with open('model/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("Metrik disimpan ke model/metrics.json")

# ============================================================
# 9. CONFUSION MATRIX PLOT
# ============================================================
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoders['NObeyesdad'].classes_,
    yticklabels=label_encoders['NObeyesdad'].classes_
)
plt.title('Confusion Matrix — KNN Optimized (Test Set)')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('model/confusion_matrix.png', dpi=150)
plt.close()
print("Confusion matrix disimpan ke model/confusion_matrix.png")

# ============================================================
# 10. SIMPAN PIPELINE (model + scaler + selector + encoders)
# ============================================================
pipeline = {
    "model"    : best_knn,
    "scaler"   : scaler,
    "selector" : selector,
    "encoders" : label_encoders
}
joblib.dump(pipeline, 'model/pipeline.pkl')

# Simpan juga file individual untuk kompatibilitas
joblib.dump(best_knn,      'model/knn_model.pkl')
joblib.dump(scaler,        'model/scaler.pkl')
joblib.dump(selector,      'model/selector.pkl')
joblib.dump(label_encoders,'model/encoder.pkl')

print("\nSemua komponen pipeline berhasil disimpan!")
print(f"  Akurasi Final (Test) : {test_accuracy*100:.2f}%")