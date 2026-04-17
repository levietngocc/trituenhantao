import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ==========================================
# 0. HELPER: GENERATE SAMPLE DATA IF MISSING
# ==========================================
def generate_sample_data(save_path="data/data.csv"):
    """Generate a realistic e-commerce dataset for testing."""
    os.makedirs("data", exist_ok=True)
    np.random.seed(42)
    n_customers = 2000
    n_invoices = 8000

    customer_ids = np.arange(10000, 10000 + n_customers)
    countries = ['United Kingdom', 'Germany', 'France', 'USA', 'Australia', 'Spain']
    descriptions = [
        'White Hanging Heart T-Light Holder', 'Jumbo Bag Red Retrospot',
        'Regency Cakestand 3 Tier', 'Wooden Heart Box', 'Vintage Union Jack Tea Set',
        'Hand Warmer Union Jack', 'Set Of 12 Wooden Clothes Pegs', 'Cream Mug With Heart',
        'Red Retrospot Tea Plate', 'Blue Hanging Heart Decoration'
    ]

    data = []
    for inv_id in range(n_invoices):
        cust = np.random.choice(customer_ids)
        country = np.random.choice(countries, p=[0.7, 0.1, 0.08, 0.05, 0.04, 0.03])
        invoice_date = datetime(2022, 1, 1) + timedelta(days=np.random.randint(0, 600))
        inv_no = f"INV-{inv_id+540000}"
        n_lines = np.random.randint(1, 6)
        for _ in range(n_lines):
            qty = np.random.randint(1, 20)
            price = round(np.random.uniform(0.5, 50), 2)
            total = qty * price
            desc = np.random.choice(descriptions)
            data.append([inv_no, desc, qty, invoice_date, price, cust, country, total])

    df = pd.DataFrame(data, columns=[
        'InvoiceNo', 'Description', 'Quantity', 'InvoiceDate',
        'UnitPrice', 'CustomerID', 'Country', 'TotalPrice'
    ])
    df.to_csv(save_path, index=False, encoding='latin1')
    return df

# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="CRM Intelligence — E‑Commerce",
    layout="wide",
    page_icon="🛍️",
    initial_sidebar_state="expanded"
)

custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,600;14..32,700&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg-primary:    #0a0c10;
        --bg-card:       #111318;
        --bg-card2:      #161a22;
        --accent-cyan:   #00e5ff;
        --accent-green:  #69ffb4;
        --accent-purple: #bf93ff;
        --accent-orange: #ff9f43;
        --accent-red:    #ff6b8a;
        --text-primary:  #e8edf5;
        --text-muted:    #636e87;
        --border:        rgba(255,255,255,0.07);
    }

    html, body, .stApp {
        background: var(--bg-primary) !important;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif !important;
    }

    [data-testid="stSidebar"] {
        background: #0d0f14 !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: var(--text-primary) !important;
    }

    div[data-testid="metric-container"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border);
        border-radius: 20px !important;
        padding: 20px 16px !important;
        transition: transform 0.2s ease, border-color 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        border-color: var(--accent-cyan);
        box-shadow: 0 15px 30px rgba(0,0,0,0.4);
    }
    div[data-testid="stMetricValue"] {
        color: var(--accent-cyan) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--text-muted) !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-card);
        padding: 6px;
        border-radius: 16px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--text-muted) !important;
        border-radius: 12px !important;
        padding: 8px 20px !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 229, 255, 0.12) !important;
        color: var(--accent-cyan) !important;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent-green);
        margin: 28px 0 16px;
        padding-left: 12px;
        border-left: 3px solid var(--accent-green);
    }
    .section-title-cyan {
        color: var(--accent-cyan);
        border-left-color: var(--accent-cyan);
    }

    .insight-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 12px;
    }

    .stDownloadButton button {
        background: linear-gradient(135deg, #00e5ff22, #69ffb422) !important;
        color: var(--accent-green) !important;
        font-weight: 600;
        border: 1px solid var(--accent-green) !important;
        border-radius: 12px !important;
        transition: all 0.2s;
    }
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #00e5ff44, #69ffb444) !important;
        box-shadow: 0 0 15px rgba(105, 255, 180, 0.3);
        transform: translateY(-2px);
    }

    hr {
        border-color: var(--border);
        margin: 20px 0;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem !important; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# CẤU HÌNH FONT CHO MATPLOTLIB (HỖ TRỢ TIẾNG VIỆT)
# ==========================================
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor':   '#111318',
    'axes.edgecolor':   '#2a2f3d',
    'axes.labelcolor':  '#9aa3b8',
    'xtick.color':      '#636e87',
    'ytick.color':      '#636e87',
    'grid.color':       '#1e2330',
    'text.color':       '#e8edf5',
})

PALETTE = ['#00e5ff', '#69ffb4', '#bf93ff', '#ff9f43', '#ff6b8a', '#ffd166']

# ==========================================
# 2. HEADER
# ==========================================
col_logo, col_title = st.columns([1, 9])
with col_title:
    st.markdown("""
    <div style="padding: 6px 0 4px;">
        <div style="font-size:0.7rem; letter-spacing:0.3em; color:#636e87; text-transform:uppercase;">
            🧠 Machine Learning · RFM · Customer Intelligence
        </div>
        <div style="font-size:2.5rem; font-weight:700; letter-spacing:-0.02em; line-height:1.2;
                    background: linear-gradient(90deg, #00e5ff 0%, #69ffb4 60%, #bf93ff 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            E‑COMMERCE CRM INTELLIGENCE
        </div>
        <div style="font-size:0.85rem; color:#636e87; margin-top:6px;">
            Phân tích RFM · Phân cụm khách hàng bằng K‑Means · Đề xuất chiến lược cá nhân hóa
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ==========================================
# 3. DATA LOADING (with fallback)
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    file_path = 'data/data.csv'
    if not os.path.exists(file_path):
        st.warning("⚠️ Không tìm thấy file 'data/data.csv'. Đang tạo dữ liệu mẫu để demo...")
        df = generate_sample_data(file_path)
    else:
        df = pd.read_csv(file_path, encoding='latin1')

    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')

    max_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg(
        Recency  = ('InvoiceDate', lambda x: (max_date - x.max()).days),
        Frequency= ('InvoiceNo',   'nunique'),
        Monetary = ('TotalPrice',  'sum'),
    ).rename(columns={
        'Recency':   'Ngày_cách_lần_cuối',
        'Frequency': 'Số_lần_mua',
        'Monetary':  'Tổng_chi_tiêu',
    })

    rfm_log = np.log1p(rfm)
    scaler  = RobustScaler()
    X_scaled = scaler.fit_transform(rfm_log)
    return df, rfm, X_scaled, scaler

with st.spinner('⏳ Đang tải và xử lý dữ liệu...'):
    df_raw, rfm_df, X_scaled, scaler = load_and_preprocess_data()

# ==========================================
# 4. SIDEBAR – CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 0 6px;'>
        <div style='font-size:1rem; font-weight:700; color:#00e5ff; letter-spacing:0.05em;'>⚙️ CẤU HÌNH MÔ HÌNH</div>
    </div>
    """, unsafe_allow_html=True)

    k_clusters = st.slider(
        "Số cụm khách hàng (K)",
        min_value=2, max_value=8, value=4, step=1,
        help="K‑Means sẽ phân chia khách hàng thành K nhóm dựa trên hành vi mua sắm."
    )

    st.markdown("---")
    st.markdown("<div style='font-size:0.8rem; font-weight:600; color:#636e87;'>🎯 LỌC DỮ LIỆU</div>", unsafe_allow_html=True)

    all_countries = sorted(df_raw['Country'].dropna().unique().tolist())
    selected_countries = st.multiselect(
        "Quốc gia",
        all_countries,
        default=['United Kingdom'],
        help="Chỉ phân tích khách hàng từ các quốc gia được chọn."
    )

    min_spend = st.number_input(
        "Chi tiêu tối thiểu (£)",
        min_value=0, value=0, step=50,
        help="Chỉ giữ lại khách hàng có tổng chi tiêu >= ngưỡng này."
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#3d4560; line-height:1.6;'>
        <b>Thuật toán:</b> K‑Means<br>
        <b>Tiền xử lý:</b> log(1+x) + RobustScaler<br>
        <b>Trực quan:</b> PCA (2 thành phần chính)<br>
        <b>Chỉ số:</b> RFM (Recency, Frequency, Monetary)
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. APPLY FILTERS & RUN CLUSTERING
# ==========================================
df_filtered = df_raw.copy()
if selected_countries:
    df_filtered = df_filtered[df_filtered['Country'].isin(selected_countries)]

@st.cache_data
def compute_rfm_filtered(df_f, min_s):
    max_date = df_f['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df_f.groupby('CustomerID').agg(
        Recency  = ('InvoiceDate', lambda x: (max_date - x.max()).days),
        Frequency= ('InvoiceNo',   'nunique'),
        Monetary = ('TotalPrice',  'sum'),
    ).rename(columns={
        'Recency':   'Ngày_cách_lần_cuối',
        'Frequency': 'Số_lần_mua',
        'Monetary':  'Tổng_chi_tiêu',
    })
    rfm = rfm[rfm['Tổng_chi_tiêu'] >= min_s]
    if len(rfm) == 0:
        return rfm, None
    rfm_log = np.log1p(rfm)
    scaler   = RobustScaler()
    Xs       = scaler.fit_transform(rfm_log)
    return rfm, Xs

rfm_work, Xs = compute_rfm_filtered(df_filtered, min_spend)

if rfm_work is None or len(rfm_work) == 0:
    st.error("❌ Không có khách hàng nào thỏa mãn điều kiện lọc. Vui lòng giảm ngưỡng chi tiêu hoặc chọn quốc gia khác.")
    st.stop()

# K-Means for main tabs
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
rfm_work['Mã_nhóm'] = kmeans.fit_predict(Xs)

# Identify VIP and Churn clusters
cluster_means = rfm_work.groupby('Mã_nhóm').mean()
vip_cluster   = cluster_means['Tổng_chi_tiêu'].idxmax()
churn_cluster = cluster_means['Ngày_cách_lần_cuối'].idxmax()

CLUSTER_META = {
    vip_cluster:   {"label": "💎 VIP (Chi tiêu cao nhất)",     "color": PALETTE[0], "short": "VIP"},
    churn_cluster: {"label": "⚠️ Nguy cơ rời bỏ (Recency cao)", "color": PALETTE[4], "short": "Churn"},
}
def get_meta(cid):
    return CLUSTER_META.get(cid, {"label": f"⭐ Nhóm {cid} (Tiềm năng)", "color": PALETTE[cid % len(PALETTE)], "short": f"N{cid}"})

rfm_work['Phân_loại'] = rfm_work['Mã_nhóm'].apply(lambda c: get_meta(c)["label"])
rfm_work['Màu_nhóm']  = rfm_work['Mã_nhóm'].apply(lambda c: get_meta(c)["color"])

# ==========================================
# 6. TABS (5 tabs)
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  TỔNG QUAN",
    "🤖  BẢN ĐỒ PHÂN CỤM (PCA)",
    "🔍  HỒ SƠ TỪNG CỤM",
    "🎯  CHIẾN LƯỢC & XUẤT DỮ LIỆU",
    "📈  SO SÁNH THUẬT TOÁN (ĐỒ ÁN)"
])

# ---------- TAB 1: TỔNG QUAN (giữ nguyên) ----------
with tab1:
    st.markdown('<div class="section-title">📈 CHỈ SỐ KPI TOÀN HỆ THỐNG</div>', unsafe_allow_html=True)

    total_rev  = df_filtered['TotalPrice'].sum()
    total_ord  = df_filtered['InvoiceNo'].nunique()
    total_cust = rfm_work.shape[0]
    aov        = total_rev / total_ord if total_ord else 0
    avg_freq   = rfm_work['Số_lần_mua'].mean()
    avg_rec    = rfm_work['Ngày_cách_lần_cuối'].mean()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("💰 Doanh thu",        f"£{total_rev:,.0f}")
    k2.metric("📦 Số đơn hàng",      f"{total_ord:,}")
    k3.metric("👥 Khách hàng",       f"{total_cust:,}")
    k4.metric("🛒 Giá trị đơn hàng TB", f"£{aov:,.1f}")
    k5.metric("🔁 Tần suất mua TB",    f"{avg_freq:.1f} đơn")
    k6.metric("📅 Số ngày kể từ lần cuối (TB)", f"{avg_rec:.0f} ngày")

    st.markdown('<div class="section-title">📆 DOANH THU THEO THÁNG & 🏆 TOP SẢN PHẨM</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        monthly = (df_filtered.groupby('YearMonth')['TotalPrice']
                   .sum().reset_index().sort_values('YearMonth'))
        monthly['YM_str'] = monthly['YearMonth'].astype(str)

        fig_rev, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(range(len(monthly)), monthly['TotalPrice'], alpha=0.2, color='#00e5ff')
        ax.plot(range(len(monthly)), monthly['TotalPrice'], color='#00e5ff', linewidth=2.5,
                marker='o', markersize=5, markerfacecolor='#0a0c10', markeredgecolor='#00e5ff')
        max_idx = monthly['TotalPrice'].idxmax()
        ax.annotate(f"£{monthly.loc[max_idx,'TotalPrice']:,.0f}",
                    xy=(max_idx - monthly.index[0], monthly.loc[max_idx,'TotalPrice']),
                    xytext=(0, 12), textcoords='offset points', ha='center',
                    color='#69ffb4', fontsize=8, fontweight='bold')
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels(monthly['YM_str'], rotation=45, ha='right', fontsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'£{x/1000:.0f}K'))
        ax.set_ylabel('Doanh thu (£)')
        ax.set_title('Doanh thu theo tháng', color='#e8edf5', fontsize=11, fontweight='600')
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        fig_rev.tight_layout()
        st.pyplot(fig_rev)

    with col_right:
        top_prod = (df_filtered.groupby('Description')['TotalPrice']
                    .sum().nlargest(10).sort_values())
        fig_prod, ax2 = plt.subplots(figsize=(7, 4.2))
        colors_bar = [PALETTE[i % len(PALETTE)] for i in range(len(top_prod))]
        bars = ax2.barh(range(len(top_prod)), top_prod.values, color=colors_bar, height=0.65)
        ax2.set_yticks(range(len(top_prod)))
        ax2.set_yticklabels([t[:30] + '…' if len(t) > 30 else t for t in top_prod.index], fontsize=7)
        for i, (bar, val) in enumerate(zip(bars, top_prod.values)):
            ax2.text(val * 1.01, i, f'£{val/1000:.1f}K', va='center', fontsize=7, color='#9aa3b8')
        ax2.set_xlabel('Doanh thu (£)')
        ax2.set_title('Top 10 sản phẩm theo doanh thu', color='#e8edf5', fontsize=11, fontweight='600')
        ax2.spines[['top','right','left','bottom']].set_visible(False)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'£{x/1000:.0f}K'))
        fig_prod.tight_layout()
        st.pyplot(fig_prod)

    st.markdown('<div class="section-title">📄 100 GIAO DỊCH GẦN NHẤT</div>', unsafe_allow_html=True)
    display_df = df_filtered[['InvoiceNo','CustomerID','Country','InvoiceDate','Quantity','TotalPrice']].head(100).copy()
    display_df.columns = ['Mã đơn', 'Khách hàng', 'Quốc gia', 'Ngày mua', 'Số lượng', 'Tổng tiền (£)']
    display_df['Tổng tiền (£)'] = display_df['Tổng tiền (£)'].map('{:,.2f}'.format)
    st.dataframe(display_df, use_container_width=True, height=320)

# ---------- TAB 2: BẢN ĐỒ PCA (đã sửa lỗi font và thêm mô tả cụm) ----------
with tab2:
    st.markdown('<div class="section-title section-title-cyan">🗺️ KHÔNG GIAN 2D – PHÂN CỤM BẰNG PCA</div>', unsafe_allow_html=True)
    st.info("Mỗi chấm là một khách hàng. PCA giảm 3 chiều RFM xuống 2 chiều để trực quan. Các cụm gần nhau có hành vi mua sắm tương tự.")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(Xs)
    var_exp = pca.explained_variance_ratio_ * 100

    plot_df = pd.DataFrame({
        'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1],
        'Cluster': rfm_work['Mã_nhóm'].values,
        'Label': rfm_work['Phân_loại'].values,
        'Color': rfm_work['Màu_nhóm'].values,
    })

    # Tính các mốc trung vị để tạo mô tả
    overall_r = rfm_work['Ngày_cách_lần_cuối'].median()
    overall_f = rfm_work['Số_lần_mua'].median()
    overall_m = rfm_work['Tổng_chi_tiêu'].median()
    cluster_median = rfm_work.groupby('Mã_nhóm')[['Ngày_cách_lần_cuối', 'Số_lần_mua', 'Tổng_chi_tiêu']].median()

    col_map, col_stats = st.columns([3, 1])
    with col_map:
        fig_pca, ax = plt.subplots(figsize=(11, 7))
        for cid in sorted(plot_df['Cluster'].unique()):
            mask = plot_df['Cluster'] == cid
            meta = get_meta(cid)
            subset = plot_df[mask]
            ax.scatter(subset['PC1'], subset['PC2'], c=meta['color'], alpha=0.65, s=30,
                       edgecolors='white', linewidths=0.2, label=meta['label'])
            cx, cy = subset['PC1'].mean(), subset['PC2'].mean()
            ax.scatter(cx, cy, c=meta['color'], s=220, marker='*', edgecolors='white', linewidths=0.8)
            ax.annotate(meta['short'], (cx, cy), xytext=(6, 6), textcoords='offset points',
                        color=meta['color'], fontsize=9, fontweight='bold',
                        fontfamily='DejaVu Sans',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a0c10',
                                  edgecolor=meta['color'], alpha=0.8))
        ax.set_xlabel(f'Thành phần chính 1 ({var_exp[0]:.1f}% phương sai)', fontfamily='DejaVu Sans')
        ax.set_ylabel(f'Thành phần chính 2 ({var_exp[1]:.1f}% phương sai)', fontfamily='DejaVu Sans')
        ax.set_title(f'Phân cụm khách hàng – K = {k_clusters} nhóm', fontsize=12, fontweight='700', fontfamily='DejaVu Sans')
        ax.grid(alpha=0.15)
        ax.spines[['top','right']].set_visible(False)
        legend = ax.legend(title='Cụm:', framealpha=0.85, facecolor='#111318', edgecolor='#2a2f3d', labelcolor='white')
        legend.get_title().set_color('#00e5ff')
        legend.get_title().set_fontfamily('DejaVu Sans')
        for text in legend.get_texts():
            text.set_fontfamily('DejaVu Sans')
        fig_pca.tight_layout()
        st.pyplot(fig_pca)

    with col_stats:
        st.markdown("#### 📊 Phân bố cụm")
        cluster_sizes = rfm_work['Mã_nhóm'].value_counts().sort_index()
        total_c = cluster_sizes.sum()
        for cid, cnt in cluster_sizes.items():
            meta = get_meta(cid)
            pct = cnt / total_c * 100
            if cid == vip_cluster:
                desc = "🌟 KHÁCH LÕI: Chi tiêu cao nhất, mua thường xuyên. Nguồn thu chủ lực."
            elif cid == churn_cluster:
                desc = "⚠️ SẮP MẤT: Đã bỏ đi rất lâu, giá trị mang lại thấp. Nguy cơ rời bỏ rất cao."
            else:
                r_val = cluster_median.loc[cid, 'Ngày_cách_lần_cuối']
                f_val = cluster_median.loc[cid, 'Số_lần_mua']
                m_val = cluster_median.loc[cid, 'Tổng_chi_tiêu']
                traits = []
                traits.append("mới mua gần đây" if r_val < overall_r else "lâu chưa mua")
                traits.append("hay mua" if f_val > overall_f else "ít mua")
                traits.append("chi tiêu khá" if m_val > overall_m else "chi tiêu thấp")
                desc = f"🔍 TIỀM NĂNG: Khách hàng {', '.join(traits)}."
            st.markdown(f"""
            <div style='background:#111318; border-left:3px solid {meta["color"]}; border-radius:12px;
                        padding:10px; margin-bottom:10px;'>
                <div style='color:{meta["color"]}; font-weight:600;'>{meta["label"]}</div>
                <div style='font-size:1.4rem; font-weight:700;'>{cnt:,}</div>
                <div style='font-size:0.7rem; color:#636e87;'>{pct:.1f}% khách hàng</div>
                <div style='font-size:0.7rem; margin-top:8px; color:#9aa3b8;'>{desc}</div>
                <div style='background:#1e2330; border-radius:4px; height:4px; margin-top:6px;'>
                    <div style='background:{meta["color"]}; width:{pct:.1f}%; height:4px; border-radius:4px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Elbow curve
    st.markdown('<div class="section-title section-title-cyan">📐 ELBOW CURVE – CHỌN SỐ CỤM TỐI ƯU</div>', unsafe_allow_html=True)
    @st.cache_data
    def compute_inertias(Xs_array, max_k=9):
        inertias = []
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init='auto')
            km.fit(Xs_array)
            inertias.append(km.inertia_)
        return inertias
    inertias = compute_inertias(Xs, max_k=9)
    ks = list(range(2, len(inertias)+2))
    fig_elbow, ax_e = plt.subplots(figsize=(9, 3.5))
    ax_e.plot(ks, inertias, color='#bf93ff', linewidth=2, marker='o', markersize=6,
              markerfacecolor='#0a0c10', markeredgecolor='#bf93ff')
    ax_e.axvline(x=k_clusters, color='#69ffb4', linestyle='--', linewidth=1.5, alpha=0.7, label=f'K hiện tại = {k_clusters}')
    ax_e.fill_between(ks, inertias, alpha=0.1, color='#bf93ff')
    ax_e.set_xlabel('Số cụm K', fontfamily='DejaVu Sans')
    ax_e.set_ylabel('Inertia (tổng bình phương nội cụm)', fontfamily='DejaVu Sans')
    ax_e.set_title('Elbow method – tìm K tối ưu', fontsize=11, fontweight='600', fontfamily='DejaVu Sans')
    ax_e.legend()
    ax_e.grid(alpha=0.2)
    ax_e.spines[['top','right']].set_visible(False)
    fig_elbow.tight_layout()
    st.pyplot(fig_elbow)

# ---------- TAB 3: HỒ SƠ CỤM (đã sửa lỗi font và đổi tên cột) ----------
with tab3:
    st.markdown('<div class="section-title">📋 HỒ SƠ CHI TIẾT TỪNG NHÓM KHÁCH HÀNG</div>', unsafe_allow_html=True)

    # 1. BỔ SUNG LỜI GIẢI THÍCH VÀ ĐẶC ĐIỂM CHÂN DUNG TỪNG NHÓM
   

    # Tính toán trung vị chung của toàn hệ thống để làm mốc so sánh
    overall_r = rfm_work['Ngày_cách_lần_cuối'].median()
    overall_f = rfm_work['Số_lần_mua'].median()
    overall_m = rfm_work['Tổng_chi_tiêu'].median()

    summary = (rfm_work
               .groupby(['Mã_nhóm', 'Phân_loại'])
               [['Ngày_cách_lần_cuối', 'Số_lần_mua', 'Tổng_chi_tiêu']]
               .agg(['mean', 'median'])
               .reset_index())
    
    summary.columns = ['Mã nhóm', 'Phân loại', 'Recency TB', 'Recency trung vị',
                       'Frequency TB', 'Frequency trung vị', 'Monetary TB', 'Monetary trung vị']
    # Đổi tên cột TB -> Trung bình
    summary.rename(columns={
        'Recency TB': 'Recency Trung bình',
        'Frequency TB': 'Frequency Trung bình',
        'Monetary TB': 'Monetary Trung bình'
    }, inplace=True)
    summary['Số khách'] = rfm_work.groupby('Mã_nhóm').size().values
    summary['% Tổng'] = (summary['Số khách'] / len(rfm_work) * 100).round(1).astype(str) + '%'

    # Tạo cột nhận định (Insight) tự động cho từng nhóm
    insights = []
    for _, row in summary.iterrows():
        cid = row['Mã nhóm']
        if cid == vip_cluster:
            insights.append("🌟 KHÁCH LÕI: Chi tiêu cao nhất, mua thường xuyên. Nguồn thu chủ lực.")
        elif cid == churn_cluster:
            insights.append("⚠️ SẮP MẤT: Đã bỏ đi rất lâu, giá trị mang lại thấp. Nguy cơ rời bỏ rất cao.")
        else:
            r_val = row['Recency trung vị']
            f_val = row['Frequency trung vị']
            m_val = row['Monetary trung vị']
            
            traits = []
            traits.append("mới mua gần đây" if r_val < overall_r else "lâu chưa mua")
            traits.append("hay mua" if f_val > overall_f else "ít mua")
            traits.append("chi tiêu khá" if m_val > overall_m else "chi tiêu thấp")
            
            insights.append(f"🔍 TIỀM NĂNG: Khách hàng {', '.join(traits)}.")
            
    summary['Đặc điểm nhận diện'] = insights

    # 2. ĐỊNH DẠNG LẠI DATAFRAME ĐỂ CẮT BỎ CÁC SỐ 0 DƯ THỪA
    format_dict = {
        'Recency Trung bình': '{:.0f} ngày', 
        'Recency trung vị': '{:.0f} ngày',
        'Frequency Trung bình': '{:.1f}', 
        'Frequency trung vị': '{:.0f}',
        'Monetary Trung bình': '£{:,.0f}', 
        'Monetary trung vị': '£{:,.0f}'
    }
    
    # Sắp xếp lại thứ tự cột cho hợp lý
    cols_order = ['Mã nhóm', 'Phân loại', 'Đặc điểm nhận diện', 'Số khách', '% Tổng', 
                  'Recency trung vị', 'Frequency trung vị', 'Monetary trung vị',
                  'Recency Trung bình', 'Frequency Trung bình', 'Monetary Trung bình']
    
    st.dataframe(
        summary[cols_order].style
                           .format(format_dict)
                           .background_gradient(cmap='Blues', subset=['Monetary Trung bình', 'Monetary trung vị']),
        use_container_width=True,
        hide_index=True
    )

    st.markdown('<div class="section-title">📊 SO SÁNH RFM (TRUNG VỊ) GIỮA CÁC CỤM</div>', unsafe_allow_html=True)
    fig_rfm, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = ['Ngày_cách_lần_cuối', 'Số_lần_mua', 'Tổng_chi_tiêu']
    titles = ['Recency (ngày – càng thấp càng tốt)', 'Frequency (số lần mua)', 'Monetary (tổng chi tiêu £)']
    
    # Hàm rút gọn hiển thị số trên biểu đồ (VD: 15200 -> 15.2K)
    def format_k(val):
        if val >= 1000:
            return f'{val/1000:.1f}K'.replace('.0K', 'K')
        return f'{val:.0f}'

    for ax, metric, title in zip(axes, metrics, titles):
        grp = rfm_work.groupby('Mã_nhóm')[metric].median().sort_index()
        colors = [get_meta(cid)['color'] for cid in grp.index]
        labels = [get_meta(cid)['short'] for cid in grp.index]
        bars = ax.bar(range(len(grp)), grp.values, color=colors, width=0.6)
        
        for i, (bar, val) in enumerate(zip(bars, grp.values)):
            # Gọi hàm format rút gọn thay vì để số nguyên cồng kềnh
            display_text = format_k(val) if metric == 'Tổng_chi_tiêu' else f'{val:.0f}'
            ax.text(i, val + grp.max()*0.02, display_text, ha='center', va='bottom', fontsize=9, color=colors[i], fontweight='bold')
            
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels(labels)
        ax.set_title(title, fontfamily='DejaVu Sans')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.grid(axis='y', alpha=0.2)
        if metric == 'Tổng_chi_tiêu':
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'£{x/1000:.0f}K'))
            
    fig_rfm.suptitle('Trung vị RFM theo từng cụm', fontsize=13, fontweight='700', y=1.02)
    fig_rfm.tight_layout()
    st.pyplot(fig_rfm)

    st.markdown('<div class="section-title">🔥 HEATMAP – ĐIỂM RFM CHUẨN HÓA (CAO = TỐT)</div>', unsafe_allow_html=True)
    heatmap_data = rfm_work.groupby('Mã_nhóm')[['Ngày_cách_lần_cuối', 'Số_lần_mua', 'Tổng_chi_tiêu']].median()
    heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-9)
    heatmap_norm['Ngày_cách_lần_cuối'] = 1 - heatmap_norm['Ngày_cách_lần_cuối']  # đảo ngược recency
    heatmap_norm.index = [get_meta(i)['label'] for i in heatmap_norm.index]
    heatmap_norm.columns = ['Recency (điểm cao = mới)', 'Frequency', 'Monetary']
    fig_heat, ax_h = plt.subplots(figsize=(9, max(2.5, k_clusters*0.7)))
    sns.heatmap(heatmap_norm, annot=True, fmt='.2f', cmap='YlGnBu',
                linewidths=1, linecolor='#0a0c10', cbar_kws={'label': 'Điểm chuẩn hóa (0→1)'},
                annot_kws={'size': 10, 'weight': 'bold', 'family': 'DejaVu Sans'}, ax=ax_h)
    ax_h.set_title('Điểm RFM chuẩn hóa – càng cao càng tốt', fontsize=12, fontweight='700', fontfamily='DejaVu Sans')
    ax_h.tick_params(colors='#9aa3b8')
    ax_h.set_yticklabels(ax_h.get_yticklabels(), fontfamily='DejaVu Sans')
    fig_heat.tight_layout()
    st.pyplot(fig_heat)

    st.markdown('<div class="section-title">📦 PHÂN PHỐI CHI TIÊU (BOXPLOT)</div>', unsafe_allow_html=True)
    fig_box, ax_b = plt.subplots(figsize=(12, 4.5))
    data_by_cluster = [rfm_work[rfm_work['Mã_nhóm'] == cid]['Tổng_chi_tiêu'].values for cid in sorted(rfm_work['Mã_nhóm'].unique())]
    colors_box = [get_meta(cid)['color'] for cid in sorted(rfm_work['Mã_nhóm'].unique())]
    labels_box = [get_meta(cid)['short'] for cid in sorted(rfm_work['Mã_nhóm'].unique())]
    bp = ax_b.boxplot(data_by_cluster, patch_artist=True, notch=False,
                      medianprops={'color': 'white', 'linewidth': 2},
                      whiskerprops={'color': '#636e87'}, capprops={'color': '#636e87'},
                      flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.4})
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor(color)
    ax_b.set_xticklabels(labels_box, fontfamily='DejaVu Sans')
    ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'£{x/1000:.0f}K'))
    ax_b.set_ylabel('Tổng chi tiêu (£)', fontfamily='DejaVu Sans')
    ax_b.set_title('Phân phối tổng chi tiêu theo cụm', fontsize=12, fontweight='700', fontfamily='DejaVu Sans')
    ax_b.grid(axis='y', alpha=0.2)
    ax_b.spines[['top','right']].set_visible(False)
    fig_box.tight_layout()
    st.pyplot(fig_box)

# ---------- TAB 4: CHIẾN LƯỢC & XUẤT (giữ nguyên) ----------
with tab4:
    st.markdown('<div class="section-title">🎯 ĐỀ XUẤT CHIẾN LƯỢC MARKETING THEO CỤM</div>', unsafe_allow_html=True)

    strategies = {
        vip_cluster: {
            "icon": "💎", "color": PALETTE[0],
            "title": "Khách hàng VIP – Duy trì & nâng cao giá trị",
            "actions": [
                "📧 Gửi email cá nhân hóa với voucher độc quyền (10% giảm giá).",
                "🎁 Chương trình tích điểm nhân đôi trong tháng.",
                "📞 Hỗ trợ ưu tiên qua hotline / Zalo riêng.",
                "🎉 Mời tham gia sự kiện flash sale trước 24h."
            ]
        },
        churn_cluster: {
            "icon": "⚠️", "color": PALETTE[4],
            "title": "Nhóm nguy cơ rời bỏ – Kích hoạt lại",
            "actions": [
                "📱 SMS tặng freeship cho đơn hàng tiếp theo.",
                "🔔 Push notification nhắc nhở sản phẩm đã xem đang giảm giá.",
                "💌 Email win‑back: khảo sát lý do + mã giảm 15%.",
                "🎯 Retargeting quảng cáo trên Facebook/Google."
            ]
        }
    }
    for cid in sorted(rfm_work['Mã_nhóm'].unique()):
        if cid not in strategies:
            strategies[cid] = {
                "icon": "⭐", "color": get_meta(cid)['color'],
                "title": f"Nhóm {cid} – Upsell & cross‑sell",
                "actions": [
                    "🛍️ Gợi ý sản phẩm bổ sung (cross‑sell) ngay trên trang.",
                    "📊 A/B test banner bundle (mua 2 giảm 20%).",
                    "🎯 Tạo Lookalike Audience trên Meta Ads.",
                    "📈 Theo dõi LTV và ưu tiên chăm sóc nhóm tăng trưởng nhanh."
                ]
            }

    for cid in sorted(strategies.keys()):
        s = strategies[cid]
        cnt = len(rfm_work[rfm_work['Mã_nhóm'] == cid])
        st.markdown(f"""
        <div style='background:#111318; border-left:4px solid {s["color"]}; border-radius:16px;
                    padding:16px 20px; margin-bottom:16px;'>
            <div style='display:flex; align-items:center; gap:10px; margin-bottom:10px;'>
                <div style='font-size:1.5rem;'>{s['icon']}</div>
                <div>
                    <div style='font-size:1rem; font-weight:700; color:{s["color"]};'>{s['title']}</div>
                    <div style='font-size:0.7rem; color:#636e87;'>{cnt:,} khách hàng ({cnt/len(rfm_work)*100:.1f}%)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        for action in s["actions"]:
            st.markdown(f"- {action}")
        st.markdown("")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📥 XUẤT DỮ LIỆU PHỤC VỤ CHIẾN DỊCH</div>', unsafe_allow_html=True)

    col_dl1, col_dl2 = st.columns(2)
    full_csv = rfm_work[['Mã_nhóm', 'Phân_loại', 'Ngày_cách_lần_cuối', 'Số_lần_mua', 'Tổng_chi_tiêu']].copy()
    full_csv.index.name = 'CustomerID'
    with col_dl1:
        st.download_button("📥 Xuất toàn bộ khách hàng (CSV)", full_csv.to_csv(index=True).encode('utf-8-sig'),
                           "CRM_all_customers.csv", mime='text/csv', use_container_width=True)
    with col_dl2:
        vip_csv = full_csv[full_csv['Mã_nhóm'] == vip_cluster].drop(columns=['Mã_nhóm'])
        st.download_button("💎 Xuất riêng khách VIP (CSV)", vip_csv.to_csv(index=True).encode('utf-8-sig'),
                           "CRM_VIP.csv", mime='text/csv', use_container_width=True)
    churn_csv = full_csv[full_csv['Mã_nhóm'] == churn_cluster].drop(columns=['Mã_nhóm'])
    st.download_button("⚠️ Xuất danh sách nguy cơ rời bỏ (CSV)", churn_csv.to_csv(index=True).encode('utf-8-sig'),
                       "CRM_churn_risk.csv", mime='text/csv', use_container_width=True)

    st.markdown("""
    <br><div style='text-align:center; color:#3d4560; font-size:0.7rem;'>
        CRM Intelligence · K‑Means + PCA · RFM Analytics
    </div>
    """, unsafe_allow_html=True)

# ---------- TAB 5: SO SÁNH THUẬT TOÁN (giữ nguyên, đã sửa font) ----------
with tab5:
    st.markdown('<div class="section-title section-title-cyan">📊 SO SÁNH CÁC THUẬT TOÁN PHÂN CỤM</div>', unsafe_allow_html=True)
    st.markdown("""
    **Yêu cầu đồ án:** So sánh ít nhất 2–3 phương pháp.  
    Dưới đây chúng tôi so sánh **5 phương pháp**:
    - **Random (baseline)**: gán nhãn ngẫu nhiên – để chứng minh dữ liệu có cấu trúc.
    - **K‑Means**: phân cụm dựa trên khoảng cách.
    - **DBSCAN**: phân cụm dựa trên mật độ, phát hiện outlier.
    - **GMM (Gaussian Mixture Model)**: mô hình xác suất, cụm dạng ellipse.
    - **Hierarchical Agglomerative**: phân cụm phân cấp.
    """)
    
    # Khởi tạo session state để lưu kết quả, tránh chạy lại nhiều lần
    if 'comparison_done' not in st.session_state:
        st.session_state.comparison_done = False
        st.session_state.df_results = None
        st.session_state.labels_dict = None
        st.session_state.last_params = None

    # Tham số cho từng thuật toán
    with st.expander("⚙️ CẤU HÌNH THAM SỐ CHO TỪNG THUẬT TOÁN", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            k_km = st.slider("K‑Means: số cụm", 2, min(8, len(rfm_work)-1), 4, key="comp_km")
            eps_db = st.number_input("DBSCAN: eps (bán kính lân cận)", 0.1, 5.0, 0.8, 0.1, key="eps_db")
            min_samples_db = st.number_input("DBSCAN: min_samples (số điểm tối thiểu)", 2, 20, 5, 1, key="min_db")
        with col2:
            n_gmm = st.slider("GMM: số thành phần", 2, min(8, len(rfm_work)-1), 4, key="comp_gmm")
            k_hier = st.slider("Hierarchical: số cụm", 2, min(8, len(rfm_work)-1), 4, key="comp_hier")
    
    current_params = (k_km, eps_db, min_samples_db, n_gmm, k_hier)
    
    # Nút chạy so sánh
    if st.button("🚀 CHẠY SO SÁNH TẤT CẢ THUẬT TOÁN", use_container_width=True):
        with st.spinner("Đang chạy các mô hình, vui lòng chờ..."):
            # Progress bar
            progress_bar = st.progress(0, text="Khởi tạo...")
            results = []
            labels_dict = {}
            
            # 1. Random Baseline
            progress_bar.progress(10, text="Random baseline...")
            np.random.seed(42)
            rand_labels = np.random.randint(0, k_km, size=len(Xs))
            sil_rand = silhouette_score(Xs, rand_labels) if len(set(rand_labels))>1 else np.nan
            db_rand = davies_bouldin_score(Xs, rand_labels) if len(set(rand_labels))>1 else np.nan
            ch_rand = calinski_harabasz_score(Xs, rand_labels) if len(set(rand_labels))>1 else np.nan
            results.append(["Random (baseline)", k_km, sil_rand, db_rand, ch_rand])
            labels_dict["Random"] = rand_labels
            
            # 2. K-Means
            progress_bar.progress(30, text="K-Means...")
            km = KMeans(n_clusters=k_km, random_state=42, n_init='auto')
            km_labels = km.fit_predict(Xs)
            results.append(["K‑Means", k_km, silhouette_score(Xs, km_labels), davies_bouldin_score(Xs, km_labels), calinski_harabasz_score(Xs, km_labels)])
            labels_dict["K‑Means"] = km_labels
            
            # 3. DBSCAN
            progress_bar.progress(50, text="DBSCAN...")
            db = DBSCAN(eps=eps_db, min_samples=min_samples_db)
            db_labels = db.fit_predict(Xs)
            n_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
            if n_db >= 2:
                results.append(["DBSCAN", n_db, silhouette_score(Xs, db_labels), davies_bouldin_score(Xs, db_labels), calinski_harabasz_score(Xs, db_labels)])
            else:
                results.append(["DBSCAN", n_db, np.nan, np.nan, np.nan])
            labels_dict["DBSCAN"] = db_labels
            
            # 4. GMM
            progress_bar.progress(70, text="GMM...")
            gmm = GaussianMixture(n_components=n_gmm, random_state=42)
            gmm_labels = gmm.fit_predict(Xs)
            results.append(["GMM", n_gmm, silhouette_score(Xs, gmm_labels), davies_bouldin_score(Xs, gmm_labels), calinski_harabasz_score(Xs, gmm_labels)])
            labels_dict["GMM"] = gmm_labels
            
            # 5. Hierarchical
            progress_bar.progress(90, text="Hierarchical...")
            hier = AgglomerativeClustering(n_clusters=k_hier)
            hier_labels = hier.fit_predict(Xs)
            results.append(["Hierarchical", k_hier, silhouette_score(Xs, hier_labels), davies_bouldin_score(Xs, hier_labels), calinski_harabasz_score(Xs, hier_labels)])
            labels_dict["Hierarchical"] = hier_labels
            
            progress_bar.progress(100, text="Hoàn tất!")
            progress_bar.empty()
            
            # Lưu vào session state
            st.session_state.df_results = pd.DataFrame(results, columns=["Thuật toán", "Số cụm", "Silhouette ↑", "Davies‑Bouldin ↓", "Calinski‑Harabasz ↑"])
            st.session_state.labels_dict = labels_dict
            st.session_state.comparison_done = True
            st.session_state.last_params = current_params
    
    # Hiển thị kết quả nếu đã có
    if st.session_state.comparison_done and st.session_state.df_results is not None:
        df_res = st.session_state.df_results
        st.dataframe(df_res.style.highlight_max(subset=["Silhouette ↑", "Calinski‑Harabasz ↑"], color='lightgreen')
                                .highlight_min(subset=["Davies‑Bouldin ↓"], color='lightcoral'), use_container_width=True)
        
        # Biểu đồ Silhouette
        fig, ax = plt.subplots(figsize=(10,5))
        valid = df_res.dropna(subset=["Silhouette ↑"])
        if len(valid) > 0:
            colors = [PALETTE[i%len(PALETTE)] for i in range(len(valid))]
            ax.bar(valid["Thuật toán"], valid["Silhouette ↑"], color=colors)
            ax.set_ylabel("Silhouette Score", fontfamily='DejaVu Sans')
            ax.set_title("So sánh Silhouette giữa các thuật toán (cao hơn = tốt hơn)", fontfamily='DejaVu Sans')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=15)
            st.pyplot(fig)
        else:
            st.warning("Không đủ dữ liệu để vẽ biểu đồ Silhouette.")
        
        # Phân tích lỗi (error analysis)
        st.markdown("---")
        st.markdown("### 🔍 Phân tích lỗi – Khách hàng có silhouette thấp nhất")
        st.markdown("Những khách hàng có điểm silhouette thấp thường nằm ở ranh giới giữa các cụm, có thể do hành vi bất thường hoặc dữ liệu nhiễu.")
        
        chosen_algo = st.selectbox("Chọn thuật toán để xem chi tiết", list(st.session_state.labels_dict.keys()), key="error_algo")
        if chosen_algo:
            labels_err = st.session_state.labels_dict[chosen_algo]
            if len(set(labels_err)) > 1:
                sil_samples = silhouette_samples(Xs, labels_err)
                df_err = rfm_work.copy()
                df_err["Silhouette_ca_nhan"] = sil_samples
                df_err["Nhom_duoc_gan"] = labels_err
                worst = df_err.nsmallest(10, "Silhouette_ca_nhan")
                st.write("🔻 **10 khách hàng có silhouette thấp nhất (ranh giới mơ hồ):**")
                st.dataframe(worst[["Ngày_cách_lần_cuối", "Số_lần_mua", "Tổng_chi_tiêu", "Nhom_duoc_gan", "Silhouette_ca_nhan"]], use_container_width=True)
                st.write("📊 **Phân bố số lượng khách hàng theo cụm:**")
                st.bar_chart(df_err["Nhom_duoc_gan"].value_counts())
            else:
                st.warning("Thuật toán này chỉ tạo ra 1 cụm hoặc toàn bộ là noise, không thể phân tích silhouette.")
        
        # Ablation study
        st.markdown("---")
        st.markdown("### 🧪 Ablation Study – Ảnh hưởng của bước log1p")
        st.markdown("Chúng tôi thử nghiệm K‑Means với cùng số cụm nhưng **không sử dụng log1p** (chỉ scale thông thường) để xem sự khác biệt.")
        if st.button("Chạy Ablation (bỏ log1p)", key="ablation_btn"):
            with st.spinner("Đang chạy ablation..."):
                rfm_no_log = rfm_work[['Ngày_cách_lần_cuối', 'Số_lần_mua', 'Tổng_chi_tiêu']].copy()
                scaler_no_log = RobustScaler()
                Xs_no_log = scaler_no_log.fit_transform(rfm_no_log)
                km_no_log = KMeans(n_clusters=k_km, random_state=42, n_init='auto')
                labels_no_log = km_no_log.fit_predict(Xs_no_log)
                sil_no_log = silhouette_score(Xs_no_log, labels_no_log)
                db_no_log = davies_bouldin_score(Xs_no_log, labels_no_log)
                ch_no_log = calinski_harabasz_score(Xs_no_log, labels_no_log)
                
                # Lấy kết quả có log1p từ session state
                km_labels_original = st.session_state.labels_dict["K‑Means"]
                sil_original = silhouette_score(Xs, km_labels_original)
                db_original = davies_bouldin_score(Xs, km_labels_original)
                ch_original = calinski_harabasz_score(Xs, km_labels_original)
                
                df_ablation = pd.DataFrame([
                    ["Có log1p + RobustScaler", sil_original, db_original, ch_original],
                    ["Không log1p (chỉ RobustScaler)", sil_no_log, db_no_log, ch_no_log]
                ], columns=["Phương pháp", "Silhouette ↑", "Davies‑Bouldin ↓", "Calinski‑Harabasz ↑"])
                st.dataframe(df_ablation.style.highlight_max(subset=["Silhouette ↑", "Calinski‑Harabasz ↑"], color='lightgreen')
                                           .highlight_min(subset=["Davies‑Bouldin ↓"], color='lightcoral'), use_container_width=True)
                if sil_no_log < sil_original:
                    st.success("✅ Kết luận: Bước log1p giúp cải thiện chất lượng phân cụm (Silhouette tăng).")
                else:
                    st.info("ℹ️ Với dữ liệu này, log1p không làm thay đổi đáng kể, nhưng vẫn giúp giảm ảnh hưởng của outlier chi tiêu lớn.")
    else:
        st.info("👈 Nhấn nút 'CHẠY SO SÁNH' để đánh giá các thuật toán trên dữ liệu hiện tại.")