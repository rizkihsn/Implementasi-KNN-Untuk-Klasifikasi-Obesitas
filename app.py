from flask import Flask, render_template, request, send_file
import joblib
import pandas as pd
import numpy as np
import json
import os

app = Flask(__name__)

# ============================================================
# Load pipeline: model + scaler + selector + encoders
# ============================================================
pipeline_path = os.path.join('model', 'pipeline.pkl')
if os.path.exists(pipeline_path):
    pipeline = joblib.load(pipeline_path)
    model    = pipeline['model']
    scaler   = pipeline['scaler']
    selector = pipeline['selector']   # Feature selector (SelectKBest)
    encoders = pipeline['encoders']
else:
    raise FileNotFoundError('Pipeline tidak ditemukan. Jalankan train_model.py terlebih dahulu.')

# Urutan kolom fitur (harus sama persis dengan saat training)
FEATURE_ORDER = [
    'Gender', 'Age', 'Height', 'Weight',
    'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP',
    'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
    'CALC', 'MTRANS'
]

CATEGORICAL_FEATURES = [
    'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',
    'family_history_with_overweight', 'CAEC', 'MTRANS'
]

# ============================================================
# Halaman Utama — Input Manual
# ============================================================
@app.route('/')
def home():
    return render_template('index.html')

# ============================================================
# Prediksi Manual (Form)
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = request.form['gender']
        age    = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        calc   = request.form['calc']
        favc   = request.form['favc']
        fcvc   = float(request.form['fcvc'])
        ncp    = float(request.form['ncp'])
        scc    = request.form['scc']
        smoke  = request.form['smoke']
        ch2o   = float(request.form['ch2o'])
        family = request.form['family']
        faf    = float(request.form['faf'])
        tue    = float(request.form['tue'])
        caec   = request.form['caec']
        mtrans = request.form['mtrans']

        # Encode categorical variables
        gender_enc = encoders['Gender'].transform([gender])[0]
        calc_enc   = encoders['CALC'].transform([calc])[0]
        favc_enc   = encoders['FAVC'].transform([favc])[0]
        scc_enc    = encoders['SCC'].transform([scc])[0]
        smoke_enc  = encoders['SMOKE'].transform([smoke])[0]
        family_enc = encoders['family_history_with_overweight'].transform([family])[0]
        caec_enc   = encoders['CAEC'].transform([caec])[0]
        mtrans_enc = encoders['MTRANS'].transform([mtrans])[0]

        # Susun vektor fitur sesuai FEATURE_ORDER
        # Gender, Age, Height, Weight, family, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS
        data = np.array([[gender_enc, age, height, weight, family_enc, favc_enc,
                          fcvc, ncp, caec_enc, smoke_enc, ch2o, scc_enc,
                          faf, tue, calc_enc, mtrans_enc]])

        # Step 1: Normalisasi
        data_scaled = scaler.transform(data)

        # Step 2: Feature Selection
        data_selected = selector.transform(data_scaled)

        # Step 3: Prediksi
        prediction = model.predict(data_selected)
        result = encoders['NObeyesdad'].inverse_transform(prediction)[0]
        return render_template('result.html', prediction=result)

    except Exception as e:
        return render_template('index.html', error=f'Error saat prediksi: {e}')

# ============================================================
# Upload CSV — Prediksi Batch
# ============================================================
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        if not uploaded_file:
            return render_template('upload.html', error='Tidak ada file yang dipilih.')

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            return render_template('upload.html', error=f'Gagal membaca file CSV: {e}')

        # Cek kelengkapan kolom
        missing_cols = [c for c in FEATURE_ORDER if c not in df.columns]
        if missing_cols:
            return render_template('upload.html',
                error=f'Kolom berikut tidak ditemukan di CSV: {", ".join(missing_cols)}')

        # Ambil hanya kolom fitur yang dibutuhkan (otomatis buang kolom lain)
        df_features = df[FEATURE_ORDER].copy()

        try:
            # Encode categorical columns
            for col in CATEGORICAL_FEATURES:
                df_features[col] = encoders[col].transform(df_features[col])

            # Step 1: Normalisasi
            X_scaled = scaler.transform(df_features.values)

            # Step 2: Feature Selection
            X_selected = selector.transform(X_scaled)

            # Step 3: Prediksi batch
            preds = model.predict(X_selected)
            preds_label = encoders['NObeyesdad'].inverse_transform(preds)

        except Exception as e:
            return render_template('upload.html',
                error=f'Error saat memproses data: {e}')

        results = [{'index': i, 'prediction': preds_label[i]} for i in range(len(preds_label))]
        return render_template('batch_result.html', results=results)

    return render_template('upload.html')

# ============================================================
# Halaman Metrik Model
# ============================================================
@app.route('/metrics')
def metrics_page():
    metrics_path = os.path.join('model', 'metrics.json')
    data = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data = json.load(f)
    return render_template('metrics.html',
        accuracy          = data.get('accuracy'),
        val_accuracy      = data.get('val_accuracy'),
        cv_score          = data.get('cv_score'),
        best_params       = data.get('best_params'),
        selected_features = data.get('selected_features'),
        removed_features  = data.get('removed_features'),
        train_size        = data.get('train_size'),
        val_size          = data.get('val_size'),
        test_size         = data.get('test_size')
    )

# ============================================================
# Endpoint Confusion Matrix Image
# ============================================================
@app.route('/confusion_matrix')
def confusion_matrix_image():
    path = os.path.join('model', 'confusion_matrix.png')
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    return 'Confusion matrix belum tersedia. Jalankan train_model.py terlebih dahulu.', 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)