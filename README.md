import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# KONFIGURASI HALAMAN & TEMA VISUAL GLASSMORPHISM
# ==========================================
st.set_page_config(
    page_title="Neural Mining Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism Interface)
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b132b 0%, #1c2541 50%, #0b132b 100%);
        color: #f4f6f8;
    }
    .glass-box {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .profile-card {
        background: rgba(56, 189, 248, 0.1);
        border-left: 4px solid #38bdf8;
        padding: 12px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# GENERATOR DATASET SIMULASI
# ==========================================
@st.cache_data
def generate_health_data():
    np.random.seed(42)
    sample_size = 300
    df = pd.DataFrame({
        'Glucose': np.random.normal(120, 30, sample_size).clip(70, 200),
        'BloodPressure': np.random.normal(70, 12, sample_size).clip(40, 120),
        'BMI': np.random.normal(28, 6, sample_size).clip(15, 50),
        'Age': np.random.randint(21, 80, sample_size),
        'Insulin': np.random.normal(80, 40, sample_size).clip(15, 250)
    })
    df['Outcome'] = ((df['Glucose'] > 130) & (df['BMI'] > 28) | (df['Age'] > 50)).astype(int)
    return df

@st.cache_data
def generate_geospatial_data():
    np.random.seed(101)
    total_outlets = 65
    return pd.DataFrame({
        'Store_Name': [f'Coffee Branch #{i+1}' for i in range(total_outlets)],
        'Latitude': np.random.normal(-6.2088, 0.03, total_outlets),
        'Longitude': np.random.normal(106.8456, 0.03, total_outlets),
        'Monthly_Sales': np.random.randint(600, 3800, total_outlets),
        'Competitor_Count': np.random.randint(1, 10, total_outlets)
    })

# ==========================================
# NAVIGASI SIDEBAR & PROFIL
# ==========================================
with st.sidebar:
    st.title("⚡ Neural Mining Engine")
    st.caption("Data Intelligence & Analytics Suite")
    st.markdown("---")
    
    st.markdown("### 👤 Informasi Pengembang")
    st.markdown("""
    <div class="profile-card">
        <strong style="color: #38bdf8; font-size: 1.05rem;">Syah Arifka Fizirqi</strong><br>
        <span style="color: #cbd5e1; font-size: 0.9rem;">NIM: 23146077</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    selected_option = st.radio(
        "Pilih Modul Analisis:",
        ["🩺 Medical Risk Classification", "☕ Geospatial Market Segmentation"]
    )
    st.markdown("---")

# ==========================================
# MODUL 1: MEDICAL RISK CLASSIFICATION
# ==========================================
if selected_option == "🩺 Medical Risk Classification":
    st.header("🩺 Diagnostic Risk Analytics (Diabetes)")
    st.write("Analisis prediktif berbasis Machine Learning untuk evaluasi dini risiko kesehatan pasien.")
    
    health_df = generate_health_data()
    X = health_df[['Glucose', 'BloodPressure', 'BMI', 'Age', 'Insulin']]
    y = health_df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    menu1, menu2, menu3 = st.tabs(["🎮 Simulasi Pasien", "📂 Processing Batch Data", "📊 Performa Model"])
    
    # ---------------- TAB 1: SIMULASI PATIENT ----------------
    with menu1:
        st.subheader("Simulasi Penilaian Risiko Pasien")
        col_in, col_out = st.columns([1, 1])
        
        with col_in:
            st.markdown('<div class="glass-box">', unsafe_allow_html=True)
            st.markdown("##### Parameter Klinis Pasien")
            val_glucose = st.slider("Glukosa Darah (mg/dL)", 70, 200, 115)
            val_bp = st.slider("Tekanan Darah (mmHg)", 40, 120, 75)
            val_bmi = st.slider("Indeks Massa Tubuh (BMI)", 15.0, 50.0, 25.0)
            val_age = st.slider("Usia Pasien (Tahun)", 20, 80, 32)
            val_insulin = st.slider("Kadar Insulin (mu U/ml)", 15, 250, 80)
            
            chosen_model_name = st.selectbox(
                "Pilih Algoritma Klasifikasi",
                ["K-Nearest Neighbors (KNN)", "Gaussian Naïve Bayes", "Decision Tree"]
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_out:
            st.markdown('<div class="glass-box">', unsafe_allow_html=True)
            st.markdown("##### Hasil Evaluasi Diagnostik")
            
            if chosen_model_name == "K-Nearest Neighbors (KNN)":
                clf = KNeighborsClassifier(n_neighbors=5)
            elif chosen_model_name == "Gaussian Naïve Bayes":
                clf = GaussianNB()
            else:
                clf = DecisionTreeClassifier(random_state=42)
                
            clf.fit(X_train_scaled, y_train)
            patient_features = scaler.transform([[val_glucose, val_bp, val_bmi, val_age, val_insulin]])
            res_pred = clf.predict(patient_features)[0]
            res_prob = clf.predict_proba(patient_features)[0] if hasattr(clf, "predict_proba") else [0.5, 0.5]
            
            if res_pred == 1:
                st.error("⚠️ **Teridentifikasi Risiko Tinggi (Diabetes)**")
                st.metric(label="Probability Score", value=f"{res_prob[1]*100:.1f}%")
                st.warning("Tindakan: Direkomendasikan untuk pemeriksaan laboratorium komprehensif.")
            else:
                st.success("✅ **Teridentifikasi Risiko Rendah (Normal)**")
                st.metric(label="Probability Score", value=f"{res_prob[0]*100:.1f}%")
                st.info("Tindakan: Pertahankan pola hidup sehat dan pencegahan rutin.")
            st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TAB 2: BATCH DATA ----------------
    with menu2:
        st.subheader("Pengolahan Data Pasien Kolektif")
        file_csv = st.file_uploader("Upload berkas dataset (.CSV)", type=["csv"])
        
        if file_csv is not None:
            uploaded_df = pd.read_csv(file_csv)
            st.dataframe(uploaded_df.head(), use_container_width=True)
        else:
            st.info("Visualisasi data sampel hasil inferensi batch:")
            batch_clf = KNeighborsClassifier().fit(X_train_scaled, y_train)
            sample_df = health_df.copy()
            scaled_sample = scaler.transform(sample_df[['Glucose', 'BloodPressure', 'BMI', 'Age', 'Insulin']])
            sample_df['Prediksi_Diagnosis'] = batch_clf.predict(scaled_sample)
            st.dataframe(sample_df[['Glucose', 'BMI', 'Age', 'Outcome', 'Prediksi_Diagnosis']].head(10), use_container_width=True)

    # ---------------- TAB 3: MODEL METRICS ----------------
    with menu3:
        st.subheader("Matriks Perbandingan Performa Algoritma")
        
        eval_models = {
            "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
            "Naïve Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }
        
        records = []
        for name, m in eval_models.items():
            m.fit(X_train_scaled, y_train)
            predictions = m.predict(X_test_scaled)
            records.append({
                "Algoritma": name,
                "Akurasi": f"{accuracy_score(y_test, predictions)*100:.1f}%",
                "Presisi": f"{precision_score(y_test, predictions, zero_division=0)*100:.1f}%",
                "Recall": f"{recall_score(y_test, predictions, zero_division=0)*100:.1f}%",
                "F1-Score": f"{f1_score(y_test, predictions, zero_division=0)*100:.1f}%"
            })
            
        st.table(pd.DataFrame(records))

# ==========================================
# MODUL 2: GEOSPATIAL MARKET SEGMENTATION
# ==========================================
else:
    st.header("☕ Market Segmentation & Expansion Analytics")
    st.write("Pemetaan persebaran gerai dan analisis kelayakan zonasi berbasis K-Means Clustering.")
    
    geo_df = generate_geospatial_data()
    
    left_side, right_side = st.columns([1, 2])
    
    with left_side:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        st.markdown("##### Parameter Klustering")
        k_val = st.slider("Jumlah Kluster (K)", 2, 6, 3)
        
        coords = geo_df[['Latitude', 'Longitude']]
        cluster_engine = KMeans(n_clusters=k_val, random_state=42)
        geo_df['Cluster_ID'] = cluster_engine.fit_predict(coords)
        
        st.markdown("---")
        st.markdown("##### 📍 Evaluasi Koordinat Baru")
        input_lat = st.number_input("Latitude", value=-6.2110, format="%.4f")
        input_lon = st.number_input("Longitude", value=106.8420, format="%.4f")
        
        if st.button("Uji Kelayakan Lokasi"):
            assigned_cluster = cluster_engine.predict([[input_lat, input_lon]])[0]
            st.success(f"Koordinat masuk dalam **Kluster #{assigned_cluster}**")
            
            center_point = cluster_engine.cluster_centers_[assigned_cluster]
            distance_to_center = np.sqrt((input_lat - center_point[0])**2 + (input_lon - center_point[1])**2)
            
            if distance_to_center < 0.02:
                st.warning("🔴 **Saturated Zone**: Tingkat kerapatan tinggi, potensi kompetisi ketat.")
            else:
                st.info("🔵 **Expansion Opportunity**: Zona dengan potensi penetrasi pasar baru.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with right_side:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        st.markdown("##### Peta Integritas Geospasial")
        
        center_lat = geo_df['Latitude'].mean()
        center_lon = geo_df['Longitude'].mean()
        folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB dark_matter")
        
        palette = ['#ef4444', '#3b82f6', '#10b981', '#a855f7', '#f97316', '#eab308']
        
        for _, point in geo_df.iterrows():
            cid = int(point['Cluster_ID'])
            folium.CircleMarker(
                location=[point['Latitude'], point['Longitude']],
                radius=6,
                popup=f"<b>{point['Store_Name']}</b><br>Penjualan: {point['Monthly_Sales']} unit/bln",
                color=palette[cid % len(palette)],
                fill=True,
                fill_opacity=0.8
            ).add_to(folium_map)
            
        heat_points = [[p['Latitude'], p['Longitude']] for _, p in geo_df.iterrows()]
        HeatMap(heat_points, radius=14, blur=8, opacity=0.35).add_to(folium_map)
        
        st_folium(folium_map, width="100%", height=460)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer Aplikasi
st.markdown("---")
st.caption("© 2026 Neural Mining Engine — Syah Arifka Fizirqi (23146077). Multi-Module Data Analytics Solution.")
