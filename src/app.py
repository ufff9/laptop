import os
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Laptique | Rekomendasi Laptop",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #17212b;
        --muted: #64727d;
        --line: #e6eaed;
        --accent: #0f766e;
        --accent-soft: #e6f4f1;
        --warm: #f7f4ee;
    }

    * { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink) !important; }
    .stApp { background: linear-gradient(135deg, #fbfcfb 0%, #f5f8f7 55%, #f8f5ef 100%) !important; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarContent"] { background: #f3f7f7 !important; border-right: 1px solid #dce7e5; }
    [data-testid="stSidebar"] * { color: #17212b !important; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 { color: #17212b !important; }
    [data-testid="stSidebar"] .stCaption { color: #64727d !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div { background: #24323d; border-color: #465662; }
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="input"] input { color: #17212b !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div { background: #ffffff; border-color: #ffffff; }
    [data-testid="stSidebar"] [data-baseweb="select"] [role="button"],
    [data-testid="stSidebar"] [data-baseweb="select"] [role="button"] *,
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] svg { color: #17212b !important; fill: #17212b !important; opacity: 1 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] input { color: #17212b !important; -webkit-text-fill-color: #17212b !important; }
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stMarkdownContainer"] * { color: #17212b !important; }
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"],
    [data-testid="stSidebar"] [data-testid="stSlider"] output { color: #17212b !important; opacity: 1 !important; }
    [data-testid="stSidebar"] [data-testid="stButton"] button,
    [data-testid="stSidebar"] [data-testid="stButton"] button * { background: #d8f4ed; color: #14594f !important; border: 0; }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover { background: #bdeade; }
    .block-container { max-width: 1240px; padding: 3rem 3rem 4rem; }
    .eyebrow { color: var(--accent); font-size: .76rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .55rem; }
    .hero { padding: 1.5rem 0 2.4rem; }
    main .hero h1 { color: var(--ink) !important; font-size: clamp(2.2rem, 4vw, 4.2rem); line-height: 1.02; letter-spacing: -.04em; margin: 0; max-width: 760px; }
    main .hero p { color: var(--muted) !important; font-size: 1.08rem; line-height: 1.65; max-width: 690px; margin-top: 1rem; }
    .hero-rule { height: 4px; width: 76px; background: var(--accent); border-radius: 8px; margin-top: 1.8rem; }
    .section-label { color: var(--muted); font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; margin: 1.5rem 0 .7rem; }
    .stat { background: rgba(255,255,255,.78); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.2rem; min-height: 93px; }
    .stat-label { color: var(--muted); font-size: .78rem; }
    .stat-value { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; margin-top: .35rem; }
    .result-heading { display: flex; align-items: end; justify-content: space-between; gap: 1rem; margin: 2.4rem 0 1.1rem; }
    .result-heading h2 { margin: 0; font-size: 1.65rem; letter-spacing: -.03em; }
    .result-heading p { color: var(--muted); margin: 0; }
    .laptop-card { background: rgba(255,255,255,.9); border: 1px solid var(--line); border-radius: 16px; padding: 1.35rem 1.5rem; margin: .85rem 0; box-shadow: 0 8px 24px rgba(23,33,43,.045); }
    .card-kicker { color: var(--accent); font-size: .73rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
    .card-title { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; line-height: 1.25; margin: .35rem 0 .85rem; }
    .spec { color: #50606b; font-size: .88rem; line-height: 1.7; }
    .spec strong { color: var(--ink); font-weight: 600; }
    .price { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.22rem; font-weight: 700; text-align: right; }
    .score { color: var(--accent); font-size: .78rem; font-weight: 700; text-align: right; margin-top: .45rem; }
    .empty-state { background: var(--warm); border: 1px dashed #cbd5d1; border-radius: 16px; padding: 2rem; text-align: center; margin-top: 1.3rem; }
    .empty-state h3 { margin: .2rem 0 .5rem; }
    .empty-state p { color: var(--muted); margin: 0; }
    .stButton button { border-radius: 9px; font-weight: 700; min-height: 2.8rem; }
    @media (max-width: 800px) {
        .block-container { padding: 2rem 1.2rem 3rem; }
        .hero { padding-top: .5rem; }
        .hero h1 { font-size: 2.35rem; }
        .hero p { font-size: .98rem; }
        .price, .score { text-align: left; }
    }
    @media (max-width: 640px) {
        [data-testid="stSidebar"] { min-width: 86vw; max-width: 86vw; }
        .block-container { padding: 1.3rem .9rem 2rem; }
        .hero { padding-bottom: 1.5rem; }
        .hero h1 { font-size: 2rem; }
        .hero p { line-height: 1.5; }
        .result-heading { display: block; margin-top: 1.8rem; }
        .result-heading h2 { font-size: 1.4rem; }
        .result-heading > p { margin-top: .45rem; }
        [data-testid="stHorizontalBlock"] { flex-direction: column !important; gap: .2rem !important; }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }
        .laptop-card { padding: 1.1rem; margin: .45rem 0; }
        .card-title { font-size: 1.05rem; }
        .spec { font-size: .82rem; }
        .stat { min-height: auto; padding: .8rem 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    possible_paths = [
        'laptops_cleaned_v1.csv',
        'data/laptops_cleaned_v1.csv',
        '../laptops_cleaned_v1.csv',
        '../data/laptops_cleaned_v1.csv'
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
            
    if df is None:
        st.error("File dataset 'laptops_cleaned_v1.csv' tidak ditemukan. Pastikan file berada di folder utama atau folder data.")
        st.stop()
        
    df = df.dropna(subset=['price', 'ram_num', 'memory_size', 'display_size'])
    return df

df = load_data()

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">Laptique · Personal tech guide</div>
        <h1>Temukan laptop yang terasa tepat sejak pertama dipakai.</h1>
        <p>Sesuaikan kebutuhan, anggaran, dan spesifikasi utama Anda. Mesin rekomendasi kami akan menyaring pilihan yang paling relevan untuk pekerjaan sehari-hari.</p>
        <div class="hero-rule"></div>
    </section>
    """,
    unsafe_allow_html=True,
)

# Sidebar untuk Input Preferensi Pengguna yang Lebih Lengkap
st.sidebar.markdown("## Laptique")
st.sidebar.caption("Pencari laptop yang lebih personal")
st.sidebar.markdown("### Atur preferensi")

# 1. Kategori Penggunaan (Menentukan bobot/filter awal)
use_case = st.sidebar.selectbox(
    "Tujuan Penggunaan Utama",
    ["Umum / Harian", "Gaming & Render Berat", "Programming / Development", "Kantor & Kuliah (Multitasking)"]
)

# 2. Filter Budget / Harga
max_price = st.sidebar.slider(
    "Maksimal Budget (Rp)", 
    int(df['price'].min()), 
    int(df['price'].max()), 
    int(df['price'].mean())
)

# 3. Filter Merek
selected_brand = st.sidebar.selectbox("Pilih Merek Laptop", ["Semua"] + sorted(list(df['brand_name'].unique())))

# 4. Filter Prosesor Brand (Intel vs AMD / Lainnya)
selected_proc_brand = st.sidebar.selectbox("Merek Prosesor", ["Semua"] + sorted(list(df['processor_brand'].dropna().unique())))

# 5. Filter RAM Minimal
min_ram = st.sidebar.selectbox("Minimal RAM (GB)", sorted(df['ram_num'].unique()))

# 6. Filter Penyimpanan Minimal (SSD/HDD)
min_storage = st.sidebar.selectbox("Minimal Penyimpanan (GB)", sorted(df['memory_size'].unique()))

# 7. Jenis Kartu Grafis (GPU Type)
gpu_preference = st.sidebar.selectbox("Jenis Grafis", ["Semua", "Dedicated", "Integrated"])

# Tombol Eksekusi Pencarian
if st.sidebar.button("🔍 Cari Rekomendasi Laptop"):
    # Salin dataframe untuk proses filtering
    filtered_df = df[
        (df['price'] <= max_price) & 
        (df['ram_num'] >= min_ram) & 
        (df['memory_size'] >= min_storage)
    ].copy()

    # Terapkan filter berdasarkan pilihan dropdown
    if selected_brand != "Semua":
        filtered_df = filtered_df[filtered_df['brand_name'] == selected_brand]
        
    if selected_proc_brand != "Semua":
        filtered_df = filtered_df[filtered_df['processor_brand'] == selected_proc_brand]
        
    if gpu_preference != "Semua":
        filtered_df = filtered_df[filtered_df['gpu_type'].str.lower() == gpu_preference.lower()]

    # Penyesuaian bobot otomatis berdasarkan kebutuhan penggunaan (Use Case)
    if use_case == "Gaming & Render Berat":
        # Prioritaskan GPU Dedicated dan RAM besar jika tersedia
        if 'gpu_type' in filtered_df.columns:
            filtered_df['priority_boost'] = filtered_df['gpu_type'].apply(lambda x: 1.2 if str(x).lower() == 'dedicated' else 1.0)
        else:
            filtered_df['priority_boost'] = 1.0
    elif use_case == "Programming / Development":
        # Prioritaskan RAM dan prosesor multi-core
        filtered_df['priority_boost'] = filtered_df['ram_num'].apply(lambda x: 1.2 if x >= 16 else 1.0)
    else:
        filtered_df['priority_boost'] = 1.0

    if filtered_df.empty:
        st.markdown(
            """
            <div class="empty-state">
                <h3>Belum menemukan pasangan yang cocok</h3>
                <p>Coba naikkan batas budget atau longgarkan kriteria RAM, penyimpanan, dan merek.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Content-Based Filtering menggunakan MinMaxScaler & Cosine Similarity
        scaler = MinMaxScaler()
        feature_cols = ['price', 'ram_num', 'memory_size', 'display_size', 'rating']
        
        # Pastikan kolom tersedia
        available_features = [col for col in feature_cols if col in filtered_df.columns]
        
        scaled_features = scaler.fit_transform(filtered_df[available_features].fillna(0))
        cosine_sim = cosine_similarity(scaled_features, scaled_features)
        
        # Hitung skor kemiripan dikalikan dengan boost prioritas penggunaan
        filtered_df['similarity_score'] = (cosine_sim.mean(axis=1) * 100) * filtered_df['priority_boost']
        
        # Batasi skor maksimal di 100%
        filtered_df['similarity_score'] = filtered_df['similarity_score'].clip(upper=100)

        # Urutkan berdasarkan skor kecocokan tertinggi dan rating
        recommendations = filtered_df.sort_values(by=['similarity_score', 'rating'], ascending=[False, False]).head(5)

        st.markdown(
            f"""
            <div class="result-heading">
                <div><p class="section-label">Hasil kurasi</p><h2>Rekomendasi teratas untuk Anda</h2></div>
                <p>{len(filtered_df)} laptop memenuhi filter</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        stat_cols = st.columns(3)
        stats = [
            ("Budget maksimum", f"Rp {max_price:,.0f}"),
            ("RAM minimum", f"{min_ram:g} GB"),
            ("Penyimpanan minimum", f"{min_storage:g} GB"),
        ]
        for stat_col, (label, value) in zip(stat_cols, stats):
            with stat_col:
                st.markdown(f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>', unsafe_allow_html=True)
        
        for rank, (_, row) in enumerate(recommendations.iterrows(), 1):
            col1, col2 = st.columns([3.3, 1], gap="large")
            with col1:
                st.markdown(
                    f"""
                    <div class="laptop-card">
                        <div class="card-kicker">Pilihan {rank} · {row['brand_name']}</div>
                        <div class="card-title">{row['model']}</div>
                        <div class="spec"><strong>Prosesor</strong> {row['processor']}<br><strong>Memori</strong> {row['ram']} · {row['memory_size']:g} GB {row['memory_type']}<br><strong>Grafis</strong> {row['gpu_brand']} {row['gpu_type']} · <strong>Layar</strong> {row['display_size']:g}" · <strong>Rating</strong> {row['rating']:.1f}/5</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="laptop-card">
                        <div class="stat-label">Harga perkiraan</div>
                        <div class="price">Rp {row['price']:,.0f}</div>
                        <div class="score">{row['similarity_score']:.1f}% cocok</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
else:
    st.markdown(
        """
        <div class="empty-state">
            <div class="eyebrow">Mulai pencarian</div>
            <h3>Siap menemukan pilihan Anda?</h3>
            <p>Gunakan panel di kiri untuk mengatur preferensi, lalu jalankan pencarian.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )