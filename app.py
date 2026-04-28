import streamlit as st
import pandas as pd
from pathlib import Path

from fib_pattern_engine_v2 import FibPatternEngineV2, train_and_save_model_v2

st.set_page_config(page_title='Fib Path Analyzer V2', layout='wide', page_icon='📈')

BASE_DIR = Path.cwd()
EXCEL_PATH = BASE_DIR / 'Dataset Analisis Trading.xlsx'
MODEL_PATH = BASE_DIR / 'fib_pattern_engine_v2.pkl'
FIRST_HIT_SUMMARY_PATH = BASE_DIR / 'fib_pattern_first_hit_summary_v2.csv'
REACH_SUMMARY_PATH = BASE_DIR / 'fib_pattern_reach_summary_v2.csv'

ACTIONABLE_TARGETS = ['1.61_UP', '1.61_DOWN', '2.5_UP', '2.5_DOWN', '3.6_UP', '3.6_DOWN']
FIRST_HIT_TARGETS = ACTIONABLE_TARGETS + ['TIE_SAME_BAR', 'NO_HIT_48H']
CONTINUATION_ORDER = [
    'UP_1.61_TO_2.5', 'UP_2.5_TO_3.6', 'UP_1.61_TO_3.6',
    'DOWN_1.61_TO_2.5', 'DOWN_2.5_TO_3.6', 'DOWN_1.61_TO_3.6'
]

st.title('📈 Fib Path Analyzer Dashboard V2')
st.markdown('Menampilkan **first-hit**, **reach probability semua fib**, **continuation probability**, dan **history kasus mirip**.')


@st.cache_data
def get_unique_options(excel_path: Path):
    options = {
        'trends': ['Long', 'Short'],
        'sq_moms': [],
        'sq_mom2s': [],
        'bars': [],
        'positions': ['Long', 'Short', 'No Trade'],
    }
    if not excel_path.exists():
        return options

    try:
        df = pd.read_excel(excel_path)
        for col, key in [('Trend', 'trends'), ('Squeeze Momentum', 'sq_moms'), ('Squeeze Momentum2', 'sq_mom2s')]:
            if col in df.columns:
                vals = [str(x).strip() for x in df[col].dropna().unique() if str(x).strip()]
                if vals:
                    options[key] = sorted(set(vals))

        bars = []
        for col in ['Bar 1', 'Bar 2']:
            if col in df.columns:
                bars.extend([str(x).strip() for x in df[col].dropna().unique() if str(x).strip()])
        if bars:
            options['bars'] = sorted(set(bars))

        positions = []
        for col in ['Raw Position', 'Final Position']:
            if col in df.columns:
                positions.extend([str(x).strip() for x in df[col].dropna().unique() if str(x).strip()])
        if positions:
            options['positions'] = sorted(set(positions))
    except Exception:
        pass

    return options


@st.cache_resource
def load_engine(model_path: Path):
    return FibPatternEngineV2.load(model_path)


with st.sidebar:
    st.header('⚙️ Konfigurasi Model')
    if MODEL_PATH.exists():
        st.success('✅ Model V2 ditemukan')
        if st.button('🔄 Retrain Model V2'):
            if EXCEL_PATH.exists():
                with st.spinner('Melatih ulang model V2...'):
                    train_and_save_model_v2(
                        excel_path=EXCEL_PATH,
                        model_path=MODEL_PATH,
                        first_hit_summary_csv=FIRST_HIT_SUMMARY_PATH,
                        reach_summary_csv=REACH_SUMMARY_PATH,
                    )
                load_engine.clear()
                get_unique_options.clear()
                st.success('Model V2 berhasil dilatih ulang')
                st.rerun()
            else:
                st.error('Dataset Excel tidak ditemukan')
    else:
        st.warning('⚠️ Model V2 belum ada')
        if st.button('🚀 Train Model V2'):
            if EXCEL_PATH.exists():
                with st.spinner('Training model V2...'):
                    train_and_save_model_v2(
                        excel_path=EXCEL_PATH,
                        model_path=MODEL_PATH,
                        first_hit_summary_csv=FIRST_HIT_SUMMARY_PATH,
                        reach_summary_csv=REACH_SUMMARY_PATH,
                    )
                load_engine.clear()
                get_unique_options.clear()
                st.success('Training selesai')
                st.rerun()
            else:
                st.error('Dataset Excel tidak ditemukan')

    st.markdown('---')
    st.caption('Model V2 memisahkan first-hit, reach probability semua fib, dan continuation probability.')

if not MODEL_PATH.exists():
    st.info('Silakan train model V2 dari sidebar dulu.')
    st.stop()

try:
    engine = load_engine(MODEL_PATH)
except Exception as e:
    st.error(f'Gagal load model V2: {e}')
    st.stop()

ops = get_unique_options(EXCEL_PATH)

st.header('📝 Setup Input Baru')
with st.form('input_form'):
    c1, c2, c3 = st.columns(3)
    with c1:
        trend_val = st.selectbox('Trend', options=ops['trends'] or ['Long', 'Short'])
        raw_pos_val = st.selectbox('Raw Position', options=ops['positions'] or ['Long', 'Short', 'No Trade'])
        fin_pos_val = st.selectbox('Final Position', options=ops['positions'] or ['Long', 'Short', 'No Trade'])
    with c2:
        sq_mom_val = st.selectbox('Squeeze Momentum', options=ops['sq_moms'] or ['Rise weak'])
        sq_mom2_val = st.selectbox('Squeeze Momentum2', options=ops['sq_mom2s'] or ['Rise weak'])
        bar1_val = st.selectbox('Bar 1', options=ops['bars'] or ['Red Bar Line 1'])
        bar2_val = st.selectbox('Bar 2', options=ops['bars'] or ['Red Bar Line 1'])
    with c3:
        score_val = st.number_input('Score', value=3.0, step=1.0)
        last_tr_val = st.number_input('Last TR', value=10.0, step=0.1)

    submitted = st.form_submit_button('🔮 Jalankan Prediksi', use_container_width=True)

if submitted:
    setup_data = {
        'Trend': trend_val,
        'Squeeze Momentum': sq_mom_val,
        'Squeeze Momentum2': sq_mom2_val,
        'Bar 1': bar1_val,
        'Bar 2': bar2_val,
        'Raw Position': raw_pos_val,
        'Final Position': fin_pos_val,
        'Score': score_val,
        'Last TR': last_tr_val,
    }

    with st.spinner('Menganalisis setup...'):
        result = engine.predict(setup_data, top_k_matches=5)

    st.divider()
    st.header('📊 Hasil Prediksi')

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('1️⃣ First-hit utama', result.first_hit_top_target or '-', f'{result.first_hit_top_prob:.2%}')
    m2.metric('2️⃣ Kemungkinan kedua', result.first_hit_second_target or '-', f'{result.first_hit_second_prob:.2%}')
    m3.metric('⚠️ Risk Tie Same Bar', f'{result.tie_prob:.2%}')
    m4.metric('⌚ Risk No Hit 48h', f'{result.no_hit_prob:.2%}')

    r1, r2 = st.columns(2)
    with r1:
        reach_sorted = sorted(result.reach_probs.items(), key=lambda x: x[1], reverse=True)
        top_reach = reach_sorted[0][0] if reach_sorted else '-'
        top_reach_prob = reach_sorted[0][1] if reach_sorted else 0.0
        st.metric('🎯 Reach paling mungkin', top_reach, f'{top_reach_prob:.2%}')
    with r2:
        second_reach = reach_sorted[1][0] if len(reach_sorted) > 1 else '-'
        second_reach_prob = reach_sorted[1][1] if len(reach_sorted) > 1 else 0.0
        st.metric('🎯 Reach kedua', second_reach, f'{second_reach_prob:.2%}')

    g1, g2 = st.columns(2)
    with g1:
        st.subheader('Distribusi First-hit')
        first_hit_df = pd.DataFrame({
            'Target': FIRST_HIT_TARGETS,
            'Probabilitas (%)': [result.first_hit_probs.get(k, 0.0) * 100 for k in FIRST_HIT_TARGETS]
        }).set_index('Target')
        st.bar_chart(first_hit_df)

    with g2:
        st.subheader('Reach Probability Semua Fib')
        reach_df = pd.DataFrame({
            'Target': ACTIONABLE_TARGETS,
            'Probabilitas (%)': [result.reach_probs.get(k, 0.0) * 100 for k in ACTIONABLE_TARGETS]
        }).set_index('Target')
        st.bar_chart(reach_df)

    g3, g4 = st.columns([2, 1])
    with g3:
        st.subheader('Continuation Probability')
        cont_df = pd.DataFrame({
            'Transition': CONTINUATION_ORDER,
            'Probabilitas (%)': [result.continuation_probs.get(k, 0.0) * 100 for k in CONTINUATION_ORDER]
        }).set_index('Transition')
        st.bar_chart(cont_df)

    with g4:
        st.subheader('Sumber Keputusan')
        st.metric('Exact Match', f"{result.source_summary.get('exact_match_count', 0):.0f} data")
        st.metric('Bobot Exact', f"{result.source_summary.get('exact_weight_used', 0.0):.2%}")
        st.metric('Bobot Similarity', f"{result.source_summary.get('similarity_weight_used', 0.0):.2%}")

    st.subheader('📌 Tabel Probabilitas Lengkap')
    left, right = st.columns(2)
    with left:
        first_hit_table = pd.DataFrame([
            {'Target': k, 'Probabilitas': v} for k, v in sorted(result.first_hit_probs.items(), key=lambda x: x[1], reverse=True)
        ])
        st.markdown('**First-hit probability**')
        st.dataframe(first_hit_table, use_container_width=True)
    with right:
        reach_table = pd.DataFrame([
            {'Target': k, 'Probabilitas': v} for k, v in sorted(result.reach_probs.items(), key=lambda x: x[1], reverse=True)
        ])
        st.markdown('**Reach probability semua fib**')
        st.dataframe(reach_table, use_container_width=True)

    st.markdown('**Continuation probability**')
    cont_table = pd.DataFrame([
        {'Transition': k, 'Probabilitas': v} for k, v in sorted(result.continuation_probs.items(), key=lambda x: x[1], reverse=True)
    ])
    st.dataframe(cont_table, use_container_width=True)

    st.subheader('📚 Top Kasus Historis yang Paling Mirip')
    if result.top_matches:
        matches_df = pd.DataFrame(result.top_matches)
        rename_map = {
            'date': 'Tanggal',
            'clock': 'Jam',
            'first_hit_target': 'First Hit',
            'first_hit_direction': 'Arah',
            'first_hit_level': 'Level',
            'reached_targets': 'Fib Tercapai',
            'similarity': 'Kemiripan',
            'trend': 'Trend',
            'score': 'Score',
            'last_tr': 'Last TR',
            'raw_position': 'Raw Position',
            'final_position': 'Final Position',
        }
        matches_df.rename(columns=rename_map, inplace=True)
        if 'Tanggal' in matches_df.columns:
            matches_df['Tanggal'] = matches_df['Tanggal'].astype(str).replace('NaT', '-')
        if 'Kemiripan' in matches_df.columns:
            matches_df['Kemiripan'] = matches_df['Kemiripan'].apply(lambda x: f'{x:.2%}')
        show_cols = [
            'Tanggal', 'Jam', 'First Hit', 'Arah', 'Level', 'Fib Tercapai',
            'Kemiripan', 'Trend', 'Score', 'Last TR', 'Raw Position', 'Final Position'
        ]
        show_cols = [c for c in show_cols if c in matches_df.columns]
        st.dataframe(matches_df[show_cols], use_container_width=True)
    else:
        st.info('Tidak ada data kasus historis yang cocok.')
