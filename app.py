import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Seller Performance Scoring",
    page_icon="📊",
    layout="wide"
)

# ── load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/seller_scores_final.csv')
    return df

df = load_data()

GRADE_COLORS = {
    'A': '#1D9E75',
    'B': '#378ADD',
    'C': '#BA7517',
    'D': '#E24B4A'
}

GRADE_LABELS = {
    'A': 'Premium',
    'B': 'Stable',
    'C': 'Reliable',
    'D': 'At Risk'
}

# ── sidebar ───────────────────────────────────────────────────
st.sidebar.title("Filters")

selected_grades = st.sidebar.multiselect(
    "Grade",
    options=['A', 'B', 'C', 'D'],
    default=['A', 'B', 'C', 'D'],
    format_func=lambda x: f"{x} — {GRADE_LABELS[x]}"
)

selected_states = st.sidebar.multiselect(
    "State",
    options=sorted(df['seller_state'].unique()),
    default=sorted(df['seller_state'].unique())
)

min_revenue, max_revenue = st.sidebar.slider(
    "Revenue range (R$)",
    min_value=int(df['total_revenue'].min()),
    max_value=int(df['total_revenue'].max()),
    value=(int(df['total_revenue'].min()), int(df['total_revenue'].max()))
)

# ── filter ────────────────────────────────────────────────────
filtered = df[
    (df['grade'].isin(selected_grades)) &
    (df['seller_state'].isin(selected_states)) &
    (df['total_revenue'] >= min_revenue) &
    (df['total_revenue'] <= max_revenue)
]

# ── header ────────────────────────────────────────────────────
st.title("📊 Seller Performance Scoring")
st.caption("Brazilian E-Commerce (Olist) — KMeans clustering on 1,271 sellers")

# ── metric cards ──────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Sellers", f"{len(filtered):,}")
col2.metric("Total Revenue", f"R$ {filtered['total_revenue'].sum():,.0f}")
col3.metric("Avg Review Score", f"{filtered['avg_review_score'].mean():.2f} ⭐")
col4.metric("Avg Delivery Days", f"{filtered['avg_delivery_days'].mean():.1f} days")
col5.metric("Avg On-time Rate", f"{filtered['on_time_rate'].mean():.1f}%")

st.divider()

# ── row 1: grade distribution + revenue by grade ──────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Grade distribution")
    grade_counts = filtered['grade'].value_counts().reset_index()
    grade_counts.columns = ['grade', 'count']
    grade_counts['label'] = grade_counts['grade'].map(
        lambda x: f"{x} — {GRADE_LABELS[x]}"
    )
    fig = px.pie(
        grade_counts,
        values='count',
        names='label',
        color='grade',
        color_discrete_map={
            'A': GRADE_COLORS['A'],
            'B': GRADE_COLORS['B'],
            'C': GRADE_COLORS['C'],
            'D': GRADE_COLORS['D']
        },
        hole=0.4
    )
    fig.update_layout(margin=dict(t=0, b=0), height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Revenue by grade")
    rev_by_grade = filtered.groupby('grade')['total_revenue'].mean().reset_index()
    rev_by_grade['color'] = rev_by_grade['grade'].map(GRADE_COLORS)
    fig = px.bar(
        rev_by_grade,
        x='grade',
        y='total_revenue',
        color='grade',
        color_discrete_map=GRADE_COLORS,
        labels={'total_revenue': 'Avg Revenue (R$)', 'grade': 'Grade'},
        text_auto='.0f'
    )
    fig.update_layout(showlegend=False, margin=dict(t=0, b=0), height=300)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── row 2: radar chart + scatter ──────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Grade profile — radar chart")
    radar_features = ['avg_review_score', 'on_time_rate',
                      'avg_delivery_days', 'cancellation_rate']

    radar_df = filtered.groupby('grade')[radar_features].mean()

    # нормализуем 0-1
    radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())
    # для delivery и cancellation — инвертируем (меньше = лучше)
    radar_norm['avg_delivery_days'] = 1 - radar_norm['avg_delivery_days']
    radar_norm['cancellation_rate'] = 1 - radar_norm['cancellation_rate']

    fig = go.Figure()
    for grade in radar_norm.index:
        values = radar_norm.loc[grade].tolist()
        values += values[:1]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=['Review Score', 'On-time Rate', 'Fast Delivery', 'Low Cancellation'] + ['Review Score'],
            fill='toself',
            name=f"{grade} — {GRADE_LABELS[grade]}",
            line_color=GRADE_COLORS[grade]
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        margin=dict(t=20, b=20),
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Revenue vs Review Score")
    fig = px.scatter(
        filtered,
        x='avg_review_score',
        y='total_revenue',
        color='grade',
        color_discrete_map=GRADE_COLORS,
        hover_data=['seller_id', 'seller_city', 'total_orders'],
        opacity=0.7,
        labels={
            'avg_review_score': 'Avg Review Score',
            'total_revenue': 'Total Revenue (R$)',
            'grade': 'Grade'
        }
    )
    fig.update_layout(margin=dict(t=0, b=0), height=350)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── row 3: top sellers table ──────────────────────────────────
st.subheader("Seller details")

col1, col2 = st.columns([1, 3])
with col1:
    grade_filter = st.selectbox(
        "Filter by grade",
        ['All'] + ['A', 'B', 'C', 'D']
    )

table_df = filtered if grade_filter == 'All' else filtered[filtered['grade'] == grade_filter]

st.dataframe(
    table_df[[
        'seller_id', 'seller_city', 'seller_state',
        'grade', 'segment', 'total_orders', 'total_revenue',
        'avg_review_score', 'avg_delivery_days',
        'cancellation_rate', 'on_time_rate'
    ]].sort_values('total_revenue', ascending=False),
    use_container_width=True,
    height=300,
    column_config={
        'total_revenue':      st.column_config.NumberColumn('Revenue (R$)', format='R$ %.0f'),
        'avg_review_score':   st.column_config.NumberColumn('Review ⭐', format='%.2f'),
        'avg_delivery_days':  st.column_config.NumberColumn('Delivery days', format='%.1f'),
        'cancellation_rate':  st.column_config.NumberColumn('Cancel %', format='%.2f'),
        'on_time_rate':       st.column_config.NumberColumn('On-time %', format='%.1f'),
        'grade':              st.column_config.TextColumn('Grade'),
    }
)