import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime, date

from fib_pattern_engine_v2 import FibPatternEngineV2, train_and_save_model_v2
from market_signal import compute_market_signal

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

with st.form('input_form'):
    st.subheader('🌐 Data Market (Auto-Compute)')
    m1, m2, m3 = st.columns(3)
    with m1:
        ticker_val = st.text_input('Ticker Crypto', value='ETH-USD', help='Contoh: ETH-USD, BTC-USD, SOL-USD')
    with m2:
        date_val = st.date_input('Tanggal Analisis', value=date.today())
    with m3:
        hour_val = st.slider('Jam (UTC)', min_value=0, max_value=23, value=0, help='Jam dalam UTC (0-23)')

    st.markdown('---')
    st.subheader('📊 Input Manual (Observasi Chart)')
    c1, c2 = st.columns(2)
    with c1:
        bar1_val = st.selectbox('Bar 1', options=ops['bars'] or ['Red Bar Line 1'])
    with c2:
        bar2_val = st.selectbox('Bar 2', options=ops['bars'] or ['Red Bar Line 1'])

    submitted = st.form_submit_button('🔮 Jalankan Prediksi', use_container_width=True)

if submitted:
    # ============================
    # Auto-compute Score, Last TR, Raw Position, Final Position
    # ============================
    with st.spinner('📡 Mengambil data market dan menghitung indikator...'):
        market_result = compute_market_signal(
            ticker=ticker_val,
            target_date=date_val,
            target_hour=hour_val,
        )

    # Cek error
    if market_result.get('error'):
        st.error(f"⚠️ {market_result['error']}")
        st.stop()

    score_val = market_result['score']
    last_tr_val = market_result['last_tr']
    raw_pos_val = market_result['raw_position']
    fin_pos_val = market_result['final_position']
    trend_val = market_result.get('trend', 'Long')
    sq_mom_val = market_result.get('squeeze_momentum', 'Rise weak')
    sq_mom2_val = market_result.get('squeeze_momentum2', 'Rise weak')

    # Tampilkan hasil auto-compute
    st.divider()
    st.subheader('🤖 Hasil Auto-Compute dari Market Data')

    row1 = st.columns(5)
    row1[0].metric('📈 Trend', trend_val)
    row1[1].metric('📊 Squeeze Mom', sq_mom_val)
    row1[2].metric('📊 Squeeze Mom 2', sq_mom2_val)
    row1[3].metric('Score', f'{score_val}')
    row1[4].metric('Last TR', f'{last_tr_val:.4f}')

    row2 = st.columns(2)
    row2[0].metric('Raw Position', raw_pos_val)
    row2[1].metric('Final Position', fin_pos_val)

    with st.expander('📋 Detail Indikator Market', expanded=False):
        detail_cols = st.columns(4)
        detail_cols[0].metric('Last Close', f"{market_result.get('last_close', 0):.2f}")
        detail_cols[1].metric('RSI', f"{market_result.get('rsi_last', 0):.2f}")
        detail_cols[2].metric('ADX', f"{market_result.get('adx_last', 0):.2f}")
        detail_cols[3].metric('ATR', f"{market_result.get('atr_last', 0):.4f}")

        detail_cols2 = st.columns(4)
        detail_cols2[0].metric('EMA Fast', f"{market_result.get('ema_fast_last', 0):.2f}")
        detail_cols2[1].metric('EMA Slow', f"{market_result.get('ema_slow_last', 0):.2f}")
        detail_cols2[2].metric('MACD', f"{market_result.get('macd_last', 0):.4f}")
        detail_cols2[3].metric('Filter', market_result.get('filter_reason', '-'))

    # ============================
    # Jalankan prediksi FibPatternEngineV2
    # ============================
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
