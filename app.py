import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from streamlit_folium import st_folium
import folium

# ====================================================
# 1. CONFIG HALAMAN
# ====================================================
st.set_page_config(
    page_title="UMUL AIMAN 23146039 - Enterprise Mining Engine", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================
# 2. CUSTOM CSS - ULTRA PREMIUM GEMINI GLASSMORPHISM
# ====================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1a2238 0%, #0d0e15 80%);
        color: #e3e2e6;
    }

    footer, #MainMenu {visibility: hidden;}

    [data-testid="stHeader"] {
        background-color: rgba(13, 14, 21, 0.0) !important;
    }
    [data-testid="stHeader"] button {
        color: #a8c7fa !important;
    }

    .gemini-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(168, 199, 250, 0.1) 0%, rgba(124, 172, 248, 0.05) 100%);
        border: 1px solid rgba(168, 199, 250, 0.25);
        backdrop-filter: blur(10px);
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.78rem;
        color: #a8c7fa;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .gemini-title {
        background: linear-gradient(90deg, #FFFFFF 0%, #A8C7FA 50%, #7CACF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        letter-spacing: -1px;
        margin-bottom: 6px;
    }

    .gemini-subtitle {
        color: #9aa0a6;
        font-size: 0.95rem;
        margin-bottom: 24px;
    }

    .gemini-card {
        background: rgba(23, 25, 35, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .kpi-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 16px 20px;
        text-align: center;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #a8c7fa;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #9aa0a6;
        text-transform: uppercase;
        font-weight: 600;
    }

    [data-testid="stSidebar"] {
        background-color: rgba(13, 14, 21, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    .status-positive {
        background: linear-gradient(135deg, rgba(234, 67, 53, 0.2) 0%, rgba(154, 0, 0, 0.1) 100%);
        border: 1px solid rgba(242, 184, 181, 0.4);
        color: #f2b8b5;
        padding: 18px;
        border-radius: 14px;
    }

    .status-negative {
        background: linear-gradient(135deg, rgba(52, 168, 83, 0.2) 0%, rgba(15, 81, 50, 0.1) 100%);
        border: 1px solid rgba(196, 238, 212, 0.4);
        color: #c4eed4;
        padding: 18px;
        border-radius: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# ====================================================
# 3. RESOURCE CACHING
# ====================================================
@st.cache_data
def load_dataset(filepath):
    return pd.read_csv(filepath)

@st.cache_resource
def load_ml_model(filepath):
    return joblib.load(filepath)

# ====================================================
# 4. SIDEBAR NAVIGATION
# ====================================================
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <span style="font-size: 1.8rem;">✨</span>
            <div>
                <div style="font-weight: 800; font-size: 1.1rem; color: #ffffff;">UMUL AIMAN</div>
                <div style="font-size: 0.75rem; color: #a8c7fa; font-weight: 600;">NIM: 23146039</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 0.75rem; color: #a8c7fa; font-weight: 700; letter-spacing: 1px;'>PILIH MODUL ANALISIS</p>", unsafe_allow_html=True)
    
    page = st.radio(
        "Pilih Halaman:",
        ["1. Prediksi Diabetes (Klasifikasi)", "2. Clustering Gerai Kopi (Geospasial)"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
        <div style='font-size: 0.75rem; color: #6e727e; text-align: center; line-height: 1.5;'>
            <b>Gemini Mining Engine v4.0 Enterprise</b><br>
            Powered by Scikit-Learn, Folium & Streamlit
        </div>
    """, unsafe_allow_html=True)

# ====================================================
# HALAMAN 1: KLASIFIKASI DIABETES
# ====================================================
if page == "1. Prediksi Diabetes (Klasifikasi)":
    
    st.markdown('<div class="gemini-badge">🩺 Healthcare Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-title">Prediksi Risiko Diabetes Pasien</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-subtitle">Sistem pendukung keputusan klinis dengan simulasi tunggal, analisis batch upload, dan explainable AI.</div>', unsafe_allow_html=True)

    try:
        df_diab = load_dataset('diabetes.csv')
        knn = load_ml_model('model_knn.pkl')
        nb = load_ml_model('model_nb.pkl')
        dt = load_ml_model('model_dt.pkl')
    except Exception as e:
        st.error(f"Gagal memuat dataset atau model. Pastikan file tersimpan di lokasi proyek. Detail: {e}")
        st.stop()

    target_col = next((col for col in df_diab.columns if col.lower().strip() in ['outcome', 'target', 'class', 'diabetes']), df_diab.columns[-1])
    X = df_diab.drop(target_col, axis=1)
    y = df_diab[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {'KNN': knn, 'Naïve Bayes': nb, 'Decision Tree': dt}

    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df_diab)}</div><div class="kpi-label">Total Sampel Data</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(X.columns)}</div><div class="kpi-label">Fitur Indikator</div></div>', unsafe_allow_html=True)
    with kpi3:
        best_acc = max([accuracy_score(y_test, m.predict(X_test)) for m in models.values()])
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{best_acc*100:.1f}%</div><div class="kpi-label">Akurasi Tertinggi</div></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(models)}</div><div class="kpi-label">Model Terpasang</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_sim, tab_batch, tab_eval = st.tabs(["📝 Input Tunggal", "📁 Batch Prediction (Upload File)", "📊 Analytics & Evaluasi"])

    # TAB 1: INPUT TUNGGAL
    with tab_sim:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Input Parameter Pasien")
        
        selected_model_name = st.selectbox("Algoritma Eksekusi Prediksi:", list(models.keys()), key="pred_select")
        
        feature_names = X.columns.tolist()
        input_values = []
        cols = st.columns(3)
        
        for i, col_name in enumerate(feature_names):
            default_val = float(X[col_name].median())
            min_val = float(X[col_name].min())
            max_val = float(X[col_name].max())
            
            with cols[i % 3]:
                val = st.number_input(f"{col_name}", value=default_val, min_value=min_val, max_value=max_val)
                input_values.append(val)
                
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✨ Eksekusi Analisis Prediksi"):
            chosen_model = models[selected_model_name]
            prediction = chosen_model.predict([input_values])[0]
            
            if prediction == 1:
                st.markdown("""
                    <div class="status-positive">
                        <b style="font-size: 1.05rem;">⚠️ DIAGNOSA: RISIKO DIABETES TERDETEKSI (POSITIF)</b><br>
                        <span style="font-size: 0.85rem; opacity: 0.9;">Pola indikator terdeteksi tinggi. Disarankan pemeriksaan lanjutan.</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="status-negative">
                        <b style="font-size: 1.05rem;">✅ DIAGNOSA: PASIEN DALAM KONDISI SEHAT (NEGATIF)</b><br>
                        <span style="font-size: 0.85rem; opacity: 0.9;">Profil fisiologis pasien berada dalam batas normal.</span>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2: BATCH PREDICTION
    with tab_batch:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 📁 Upload File CSV / Excel untuk Prediksi Massal")
        st.caption("Unggah dataset tanpa kolom target untuk memproses banyak pasien sekaligus.")
        
        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
        batch_model_name = st.selectbox("Model untuk Batch Processing:", list(models.keys()), key="batch_select")
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(" Preview Data Unggahan:", batch_df.head(3))
            
            if st.button("🚀 Eksekusi Prediksi Massal"):
                try:
                    chosen_batch_model = models[batch_model_name]
                    preds = chosen_batch_model.predict(batch_df[feature_names])
                    
                    batch_df['Hasil_Prediksi'] = ["Positif Diabetes" if p == 1 else "Negatif (Sehat)" for p in preds]
                    
                    st.success(" Sukses memproses seluruh data!")
                    st.dataframe(batch_df)
                    
                    # Button Download CSV
                    csv_data = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Hasil Prediksi (CSV)",
                        data=csv_data,
                        file_name="hasil_prediksi_diabetes.csv",
                        mime="text/csv"
                    )
                except Exception as err:
                    st.error(f"Kolom dalam file tidak sesuai dengan fitur model. Error: {err}")
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 3: EVALUASI & EXPLAINABILITY
    with tab_eval:
        col_left, col_right = st.columns([1.1, 0.9])
        
        with col_left:
            st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Benchmarking Performa Model")
            metrics_list = []
            for name, model in models.items():
                y_pred = model.predict(X_test)
                metrics_list.append({
                    'Algoritma': name,
                    'Akurasi': f"{accuracy_score(y_test, y_pred):.2f}",
                    'Precision': f"{precision_score(y_test, y_pred):.2f}",
                    'Recall': f"{recall_score(y_test, y_pred):.2f}",
                    'F1-Score': f"{f1_score(y_test, y_pred):.2f}"
                })
            st.dataframe(pd.DataFrame(metrics_list), use_container_width=True, hide_index=True)
            
            # Decision Tree Feature Importance
            if hasattr(dt, 'feature_importances_'):
                st.markdown("#### 🔍 Feature Importance (Decision Tree)")
                imp_df = pd.DataFrame({'Fitur': feature_names, 'Tingkat Pengaruh': dt.feature_importances_}).sort_values('Tingkat Pengaruh', ascending=True)
                fig_imp = px.bar(imp_df, x='Tingkat Pengaruh', y='Fitur', orientation='h', color='Tingkat Pengaruh', color_continuous_scale='Viridis')
                fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#e3e2e6"), height=250, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_imp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
            st.markdown("### 🧩 Confusion Matrix")
            selected_eval_model = st.selectbox("Pilih Model Evaluasi:", list(models.keys()), key="cm_select")
            
            cm = confusion_matrix(y_test, models[selected_eval_model].predict(X_test))
            fig = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Prediksi Model", y="Kondisi Real", color="Jumlah"),
                x=['Negatif (0)', 'Positif (1)'],
                y=['Negatif (0)', 'Positif (1)'],
                color_continuous_scale='Blues'
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#e3e2e6"), height=300, margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ====================================================
# HALAMAN 2: CLUSTERING GERAI KOPI
# ====================================================
elif page == "2. Clustering Gerai Kopi (Geospasial)":
    
    st.markdown('<div class="gemini-badge">☕ Spatial Intelligence Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-title">Klaster Geospasial Gerai Kopi</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-subtitle">Pengelompokan geospasial interaktif berbasis Peta Bumi Nyata (OpenStreetMap Folium).</div>', unsafe_allow_html=True)
    
    try:
        df_kopi = load_dataset('lokasi_gerai_kopi_clean.csv')
        kmeans = load_ml_model('model_kmeans.pkl')
    except Exception as e:
        st.error(f"Gagal memuat dataset/model clustering. Detail: {e}")
        st.stop()

    col_lat = [c for c in df_kopi.columns if any(k in c.lower() for k in ['lat', 'y', 'lintang'])][0]
    col_lon = [c for c in df_kopi.columns if any(k in c.lower() for k in ['lon', 'long', 'lng', 'x', 'bujur'])][0]
    
    X_kopi = df_kopi[[col_lat, col_lon]]
    df_kopi['Cluster'] = kmeans.labels_
    
    col_map, col_form = st.columns([1.3, 0.7])
    
    with col_map:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 🗺️ Peta Bumi Geospasial Interaktif")
        
        # Inisialisasi Peta Folium
        center_lat = df_kopi[col_lat].mean()
        center_lon = df_kopi[col_lon].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB dark_matter")
        
        colors = ['#FF4B4B', '#1E88E5', '#00E676', '#FFD600', '#AB47BC']
        
        # Plot Titik Gerai
        for idx, row in df_kopi.iterrows():
            cluster_id = int(row['Cluster'])
            folium.CircleMarker(
                location=[row[col_lat], row[col_lon]],
                radius=6,
                color=colors[cluster_id % len(colors)],
                fill=True,
                fill_color=colors[cluster_id % len(colors)],
                fill_opacity=0.7,
                popup=f"Gerai #{idx} (Klaster {cluster_id})"
            ).add_to(m)
            
        # Plot Centroids
        for c_idx, c_coord in enumerate(kmeans.cluster_centers_):
            folium.Marker(
                location=[c_coord[0], c_coord[1]],
                popup=f"Pusat Centroid Klaster {c_idx}",
                icon=folium.Icon(color='white', icon='star', prefix='fa')
            ).add_to(m)
            
        st_folium(m, width="100%", height=450)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_form:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Evaluasi Titik Lokasi Baru")
        
        in_lat = st.number_input(f"Latitude ({col_lat})", value=float(center_lat), format="%.6f")
        in_lon = st.number_input(f"Longitude ({col_lon})", value=float(center_lon), format="%.6f")
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📌 Analisis Karakteristik Zona"):
            pred_cluster = kmeans.predict([[in_lat, in_lon]])[0]
            
            st.markdown(f"""
                <div style="background: rgba(168, 199, 250, 0.08); padding: 14px; border-radius: 12px; margin-bottom: 16px; border: 1px solid rgba(168, 199, 250, 0.2); font-weight: 600; color: #a8c7fa;">
                    📌 Hasil Pemetaan: Tergolong dalam <b>Klaster {pred_cluster}</b>
                </div>
            """, unsafe_allow_html=True)
            
            if pred_cluster == 0:
                st.markdown("""
                    <div class="status-positive">
                        <b style="font-size: 1rem;">⚠️ STATUS ZONA: DENSITY RENDAH</b><br>
                        <span style="font-size: 0.85rem; opacity: 0.85;">Tingkat kompetisi rendah. Area ini membutuhkan strategi penetrasi pasar.</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="status-negative">
                        <b style="font-size: 1rem;">✅ STATUS ZONA: DENSITY TINGGI</b><br>
                        <span style="font-size: 0.85rem; opacity: 0.85;">Konsentrasi gerai tinggi. Menandakan foot-traffic yang sudah matang.</span>
                    </div>
                """, unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)
