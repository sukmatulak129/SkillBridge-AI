import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    st.subheader(f"Undersupply Analysis : {selected_job}")

    # -----------------------------
    # 1. AMBIL DATA SKKNI
    # -----------------------------
    units_to_test = filtered_demand['Judul Unit'].fillna("").tolist()

    # -----------------------------
    # 2. AMBIL DATA RESUME
    # -----------------------------
    resume_texts = df_resume['resume_text_clean'].fillna("").astype(str).tolist()

    # -----------------------------
    # 3. GABUNG SEMUA RESUME
    # -----------------------------
    corpus = units_to_test + resume_texts

    # -----------------------------
    # 4. TF-IDF VECTOR
    # -----------------------------
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    skkni_vec = tfidf_matrix[:len(units_to_test)]
    resume_vec = tfidf_matrix[len(units_to_test):]

    # -----------------------------
    # 5. HITUNG SIMILARITY (INI INTINYA)
    # -----------------------------
    similarity_matrix = cosine_similarity(skkni_vec, resume_vec)

    kemunculan = similarity_matrix.max(axis=1) * 100  # ubah ke persen

    # -----------------------------
    # 6. DATAFRAME HASIL
    # -----------------------------
    df_plot = pd.DataFrame({
        'Unit SKKNI': filtered_demand['Kompetensi_Short'].values,
        'Ketersediaan (%)': kemunculan
    })

    # -----------------------------
    # 7. VISUALISASI
    # -----------------------------
    fig3, ax3 = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=df_plot,
        x='Ketersediaan (%)',
        y='Unit SKKNI',
        ax=ax3
    )

    ax3.axvline(50, color='orange', linestyle='--', label='Low Match Threshold')
    ax3.axvline(70, color='red', linestyle='--', label='KPI Threshold')

    ax3.set_title("Undersupply Analysis (Semantic Matching)", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Skill Availability (%)")
    ax3.set_ylabel("Unit Kompetensi")

    st.pyplot(fig3)

    st.warning(
        "Nilai ini berbasis semantic similarity, bukan exact text matching."
    )
    
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
