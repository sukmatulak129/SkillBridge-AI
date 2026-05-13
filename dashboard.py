import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# KONFIGURASI HALAMAN
st.set_page_config( page_title="SkillBridge AI Dashboard", layout="wide")

# AUTO REFRESH (REAL-TIME SIMULATION)
st_autorefresh(interval=60000, key="dashboard_refresh")

sns.set_theme(style="whitegrid", palette="muted")

# LOAD DATA DENGAN CACHE
@st.cache_data(ttl=60)
def load_data():

    # Load dataset
    df_resume = pd.read_csv("cleaned_training_data.csv")
    df_skkni = pd.read_csv("data_pekerjaan.csv")

    # PARSE LIST SKILL
    def parse_list(x):
        try:
            return ast.literal_eval(x)
        except:
            return []

    df_resume['skills_clean_list'] = ( df_resume['skills_clean'] .apply(parse_list))

    #FREKUENSI DEMAND BERDASARKAN SKKNI
    frekuensi_demand = ( df_skkni.groupby(['Judul Unit', 'Jabatan']) .size() .reset_index(name='Frekuensi'))

    # Short title untuk visualisasi
    frekuensi_demand['Kompetensi_Short'] = (frekuensi_demand['Judul Unit'] .apply(lambda x: x[:40] + "..." if len(x) > 40 else x))

    # FREKUENSI SUPPLY DARI RESUME
    semua_skill = [
        skill.title()
        for sublist in df_resume['skills_clean_list']
        for skill in sublist
    ]

    frekuensi_supply = Counter(semua_skill)

    df_supply = pd.DataFrame( frekuensi_supply.most_common(20), columns=['Kompetensi', 'Frekuensi'])

    return df_resume, df_skkni, frekuensi_demand, df_supply

# LOAD DATA
df_resume, df_skkni, frekuensi_demand, df_supply = load_data()

# HEADER
st.title("SkillBridge AI Dashboard")
st.markdown("### Real-Time Skill Gap Analysis System")

# SIDEBAR
st.sidebar.header("Dashboard Control")

# Timestamp refresh
st.sidebar.success( f"Last Refresh:\n{datetime.now().strftime('%H:%M:%S')}")

# FILTER PEKERJAAN
list_pekerjaan = sorted( df_skkni['Jabatan'].dropna().unique().tolist())

selected_job = st.sidebar.selectbox( "Pilih Kategori Pekerjaan", ["Semua Pekerjaan"] + list_pekerjaan )

# FILTER TOP N
top_n = st.sidebar.slider( "Jumlah Top Skill", min_value=5, max_value=20, value=10 )

# MENU NAVIGASI
menu = st.sidebar.radio("Pilih Analisis",[ "Main Skill Gap", "Undersupply Analysis", "Solution Validation"])

# FILTER DATA REAL-TIME
if selected_job != "Semua Pekerjaan":
    filtered_demand = ( frekuensi_demand[frekuensi_demand['Jabatan'] == selected_job].sort_values(by='Frekuensi', ascending=False).head(top_n))

else:
    filtered_demand = ( frekuensi_demand.sort_values(by='Frekuensi', ascending=False).head(top_n))

# MAIN SKILL GAP
if menu == "Main Skill Gap":

    st.subheader(f"Skill Gap Analysis : {selected_job}")

    col1, col2 = st.columns(2)

    # VISUALISASI SUPPLY
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 6))

        sns.barplot( data=df_supply.head(top_n), x='Frekuensi', y='Kompetensi', ax=ax1, hue='Kompetensi', legend=False)

        ax1.set_title( f"Top {top_n} Skill Pelamar Internasional", fontsize=14, fontweight='bold' )
        ax1.set_xlabel("Frekuensi di Resume")
        ax1.set_ylabel("Skill")

        st.pyplot(fig1)

    # VISUALISASI DEMAND
    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))

        sns.barplot( data=filtered_demand, x='Frekuensi', y='Kompetensi_Short', ax=ax2, hue='Kompetensi_Short', legend=False )

        ax2.set_title( f"Top {top_n} Kompetensi Standar Industri", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Frekuensi Rujukan di SKKNI")
        ax2.set_ylabel("Kompetensi")

        st.pyplot(fig2)

    # INSIGHT
    st.info(
        "Dashboard menampilkan perbandingan "
        "antara supply skill pelamar dan "
        "demand kompetensi berdasarkan standar SKKNI."
    )

# UNDERSUPPLY ANALYSIS
elif menu == "Undersupply Analysis":
    st.subheader(f"Undersupply Analysis: {selected_job}")
    
    # 1. Gabungkan semua skill dari resume untuk pencarian cepat
    # Kita ambil dari kolom 'skills_clean' agar lebih akurat dibanding teks mentah
    semua_resume_skills = " ".join(df_resume['skills_clean'].astype(str).tolist()).lower()

    # 2. Ambil daftar unit dari demand yang sudah difilter
    units_to_test = filtered_demand['Judul Unit'].tolist()
    names_to_show = filtered_demand['Kompetensi_Short'].tolist()
    kemunculan = []

    # 3. Mapping Kamus Sederhana (Indo -> English)
    # Ini supaya data resume Inggris kamu bisa 'nyambung' ke SKKNI Indonesia
    mapping_kamus = {
        "keamanan": "security",
        "data": "data",
        "informasi": "information",
        "perangkat lunak": "software",
        "jaringan": "network",
        "awan": "cloud",
        "transformasi": "management",
        "industri": "business",
        "infrastruktur": "infrastructure",
        "risiko": "risk",
        "perlindungan": "protection",
        "pengujian": "testing"
    }

    for unit in units_to_test:
        unit_low = unit.lower()
        found_count = 0
        
        # Cek apakah ada kata kunci di kamus yang cocok dengan judul unit
        match_found = False
        for indo, eng in mapping_kamus.items():
            if indo in unit_low:
                # Hitung kemunculan kata Inggris-nya di resume
                found_count += semua_resume_skills.count(eng)
                match_found = True
        
        # Jika tidak ada di kamus, cari kata pertama dari judul unit (asumsi kata dasar)
        if not match_found:
            kata_kunci_asli = unit_low.split()[0]
            if len(kata_kunci_asli) > 3: # Hindari kata depan pendek seperti 'dan', 'di'
                found_count = semua_resume_skills.count(kata_kunci_asli)
        
        kemunculan.append(found_count)

    # 4. Buat DataFrame untuk Grafik
    df_plot = pd.DataFrame({
        'Unit SKKNI': names_to_show,
        'Ketersediaan di Resume': kemunculan
    })

    # 5. Visualisasi
    if df_plot['Ketersediaan di Resume'].sum() == 0:
        st.warning("⚠️ Tidak ditemukan kecocokan kata kunci yang signifikan. Hal ini menunjukkan adanya 'Language Gap' antara standar SKKNI (Indo) dan Resume (English).")
    
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_plot, x='Ketersediaan di Resume', y='Unit SKKNI', ax=ax3, palette="viridis")
    ax3.set_title("Tingkat Ketersediaan Kompetensi SKKNI pada Dataset Resume", fontsize=14, fontweight='bold')
    st.pyplot(fig3)

    st.info("Jika grafik menunjukkan angka rendah, berarti skill tersebut sangat langka (Undersupply) di pasar tenaga kerja saat ini.")
        #

# SOLUTION VALIDATION
else:
    st.subheader("Solution Validation")

    # TARGET KPI
    target_kpi = st.sidebar.number_input( "Target KPI (%)", min_value=0, max_value=100, value=70 )
    metode = ['Keyword Matching', 'Semantic Matching (AI)']
    precision_scores = [25, 75]

    # VISUALISASI
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    bars = sns.barplot( x=metode, y=precision_scores, ax=ax4, hue=metode, legend=False )

    ax4.axhline( y=target_kpi, color='red', linestyle='--', label=f'Target KPI ({target_kpi}%)' )
    ax4.set_ylim(0, 100)
    ax4.set_ylabel("Precision Score (%)")
    ax4.set_title( "Perbandingan Solusi Matching", fontsize=14, fontweight='bold' )
    ax4.legend()

    # LABEL NILAI BAR
    for bar in bars.patches:
        ax4.annotate(
            f"{int(bar.get_height())}%",
            ( bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='bottom', fontweight='bold'
        )

    st.pyplot(fig4)

    st.success(
        "Semantic Matching berbasis AI "
        "diproyeksikan memiliki performa "
        "lebih baik dibanding keyword matching."
    )

# FOOTER
st.markdown("---")
st.caption(
    "SkillBridge AI Dashboard • Real-Time Skill Gap Monitoring System"
)
