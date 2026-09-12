"""
Interview Intelligence Platform - Streamlit Web Application
Author: Placement Candidate (2026)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import recommendation_engine

# Page Configuration
st.set_page_config(
    page_title="Interview Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast CSS Styling
st.markdown("""
<style>
    /* High contrast title colors compatible with Light and Dark themes */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #2563EB;
    }
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #334155;
    }
    .xai-badge {
        background-color: #F0F9FF;
        border: 1px solid #BAE6FD;
        color: #0369A1;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 0.85rem;
        font-family: monospace;
        margin-top: 4px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return recommendation_engine.InterviewIntelligenceEngine()

engine = load_engine()

# Build Question Search Options for Interactive User Input
@st.cache_data
def get_question_options():
    options = {}
    for _, row in engine.questions.iterrows():
        qid = int(row['ID'])
        label = f"#{qid} - {row['Title']} [{row['Difficulty']}] ({row['PrimaryTopic']})"
        options[label] = qid
    return options

question_options = get_question_options()

# Sidebar Navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Module:",
    [
        "🏠 Home & Overview",
        "📊 Company Analytics",
        "🔍 Semantic Search",
        "🎯 Personalised Recommender & Dynamic Roadmap"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Multi-Objective Recommendation & NLP System")

# PAGE 1: HOME & OVERVIEW
if page == "🏠 Home & Overview":
    st.title("🎯 Interview Intelligence Platform")
    st.markdown("##### Multi-Objective Recommendation, Semantic Retrieval & Dynamic Study Roadmap Engine")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'><div class='metric-val'>3,500+</div><div class='metric-lbl'>Unique LeetCode Problems</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><div class='metric-val'>600+</div><div class='metric-lbl'>Tech Companies Tracked</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><div class='metric-val'>384-D</div><div class='metric-lbl'>Sentence Transformer Vectors</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'><div class='metric-val'>5</div><div class='metric-lbl'>Benchmarked Models</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ System Capabilities & Architecture")
    st.markdown(r"""
    This platform transforms unstructured problem statements and company interview records into a unified **384-dimensional vector space** to deliver explainable, company-aware recommendations and non-repetitive study plans.
    
    * **NLP Representation Layer:** MiniLM Sentence Transformers (`all-MiniLM-L6-v2`) encoding problem titles, tags, categories, and descriptions.
    * **Company Vector Space:** Frequency-weighted centroid vectors ($\mathbf{e}_C$) representing the interview DNA of tech companies.
    * **6-Factor Hybrid Scoring:** Combines Cosine Semantic Similarity, Frequency Bonus, Multi-Tag Topic Deficit Gap, Difficulty Alignment, and Personalization Novelty.
    * **Maximum Marginal Relevance (MMR):** Dynamic weekly roadmap scheduling that eliminates intra-topic question redundancy (executes in <50ms).
    * **Zero-Leakage Evaluation:** Strict 70/15/15 train/val/test splits per company to benchmark Recall@10, Precision@10, NDCG@10, and MAP@10.
    """)

# PAGE 2: COMPANY ANALYTICS
elif page == "📊 Company Analytics":
    st.title("📊 Company Interview Intelligence")
    st.markdown("##### Explore company-specific difficulty distributions, top yield topics, and company similarity networks.")
    
    all_companies = sorted(engine.company_features['company'].unique())
    selected_comp = st.selectbox("Select Target Company:", all_companies, index=all_companies.index('google') if 'google' in all_companies else 0)
    
    comp_row = engine.company_features[engine.company_features['company'] == selected_comp].iloc[0]
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Asked Questions", int(comp_row.get('total_questions', 0)))
    with c2:
        st.metric("Company Tier", str(comp_row.get('company_tier', 'N/A')))
    with c3:
        st.metric("Avg Acceptance Rate", f"{round(float(comp_row.get('avg_acceptance', 0)), 1)}%")
    with c4:
        st.metric("Avg Frequency Score", f"{round(float(comp_row.get('avg_frequency', 0)), 1)}%")

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("#### Difficulty Distribution")
        diff_data = pd.DataFrame({
            'Difficulty': ['Easy', 'Medium', 'Hard'],
            'Percentage': [
                float(comp_row.get('easy_pct', 0.2)),
                float(comp_row.get('medium_pct', 0.6)),
                float(comp_row.get('hard_pct', 0.2))
            ]
        })
        fig_diff = px.pie(diff_data, names='Difficulty', values='Percentage', hole=0.4, color='Difficulty', color_discrete_map={'Easy':'#10B981', 'Medium':'#F59E0B', 'Hard':'#EF4444'})
        st.plotly_chart(fig_diff, use_container_width=True)

    with ch2:
        st.markdown("#### Top Yield Topics")
        comp_topics = engine.compute_company_topic_profile(selected_comp)
        top_t_df = pd.DataFrame(list(comp_topics.items()), columns=['Topic', 'Frequency']).sort_values('Frequency', ascending=False).head(8)
        fig_topics = px.bar(top_t_df, x='Frequency', y='Topic', orientation='h', color='Frequency', color_continuous_scale='Blues')
        fig_topics.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_topics, use_container_width=True)

    st.markdown("#### Nearest Company Vector Neighbours (Semantic Embedding Space)")
    target_vec = engine.company_embeddings.get(selected_comp.lower(), np.mean(engine.embeddings, axis=0))
    sim_scores = []
    for comp_name, comp_v in engine.company_embeddings.items():
        if comp_name != selected_comp.lower():
            sim = float(cosine_similarity([target_vec], [comp_v])[0][0])
            sim_scores.append({'Company': comp_name.title(), 'Embedding Similarity': round(sim, 4)})
    sim_df = pd.DataFrame(sim_scores).sort_values('Embedding Similarity', ascending=False).head(10)
    st.dataframe(sim_df, use_container_width=True)

# PAGE 3: SEMANTIC SEARCH
elif page == "🔍 Semantic Search":
    st.title("🔍 Semantic Question Retrieval")
    st.markdown("##### Search questions by natural language concepts, topics, or problem titles.")
    
    query = st.text_input("Enter Problem Query / Concept:", "Binary search on rotated sorted array")
    top_n = st.slider("Results to Retrieve:", 5, 25, 10)
    
    if query:
        q_text_df = engine.questions.copy()
        
        # TF-IDF Match
        query_tfidf = engine.tfidf_vectorizer.transform([query])
        tfidf_sims = cosine_similarity(query_tfidf, engine.tfidf_matrix)[0]
        q_text_df['TFIDF_Score'] = tfidf_sims
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top Retrieved Questions")
            st.dataframe(
                q_text_df.sort_values('TFIDF_Score', ascending=False)[['ID', 'Title', 'Difficulty', 'PrimaryTopic', 'TFIDF_Score']].head(top_n),
                use_container_width=True
            )
            
        with c2:
            st.markdown("#### Primary Topic Breakdown")
            top_t = q_text_df.sort_values('TFIDF_Score', ascending=False)['PrimaryTopic'].head(top_n).value_counts().reset_index()
            top_t.columns = ['Topic', 'Count']
            st.dataframe(top_t, use_container_width=True)

# PAGE 4: PERSONALISED RECOMMENDER & DYNAMIC ROADMAP
elif page == "🎯 Personalised Recommender & Dynamic Roadmap":
    st.title("🎯 Personalised Recommender & Dynamic Roadmap")
    st.markdown("##### Custom recommendations and non-repetitive study plans based on your target company and solved questions.")
    
    # ------------------ USER SOLVED QUESTIONS INPUT SECTION ------------------
    st.markdown("---")
    st.markdown("### 📝 Step 1: Input Your Solved History")
    
    input_method = st.radio("Choose Input Method:", ["Search & Multi-Select Questions", "Type Question IDs", "Use Sample Profiles"], horizontal=True)
    
    solved_qids = []
    
    if input_method == "Search & Multi-Select Questions":
        selected_labels = st.multiselect(
            "Select problems you have already solved:",
            options=list(question_options.keys()),
            default=["#1 - Two Sum [Easy] (Array)", "#242 - Valid Anagram [Easy] (Hash Table)"]
        )
        solved_qids = [question_options[lbl] for lbl in selected_labels]
        
    elif input_method == "Type Question IDs":
        raw_ids_str = st.text_input("Enter comma-separated Question IDs (e.g. 1, 15, 146, 207):", "1, 15, 242")
        try:
            solved_qids = [int(i.strip()) for i in raw_ids_str.split(",") if i.strip().isdigit()]
        except:
            solved_qids = [1, 242]
            
    else: # Sample Profiles
        sample_options = {
            "Beginner Profile (Two Sum, Valid Anagram)": [1, 242],
            "Intermediate Profile (3Sum, Container Water, Level Order)": [15, 11, 102],
            "Advanced Profile (LRU Cache, Course Schedule, Merge k Lists)": [146, 207, 23]
        }
        chosen_sample = st.selectbox("Select Preset Profile:", list(sample_options.keys()))
        solved_qids = sample_options[chosen_sample]

    st.info(f"Loaded {len(solved_qids)} Solved Questions into your User Profile.")

    # ------------------ TARGET COMPANY & ACTION SECTION ------------------
    st.markdown("---")
    st.markdown("### 🏢 Step 2: Select Target Company & Mode")
    
    all_companies = sorted(engine.company_features['company'].unique())
    
    col_comp, col_mode = st.columns(2)
    with col_comp:
        target_company = st.selectbox("Select Target Company:", all_companies, index=all_companies.index('google') if 'google' in all_companies else 0)
    with col_mode:
        output_mode = st.selectbox("Select Output Generator:", ["Top-N Question Recommendations", "8-Week Progressive MMR Roadmap"])

    # MODE A: TOP-N RECOMMENDATIONS
    if output_mode == "Top-N Question Recommendations":
        top_k = st.slider("Number of Recommendations:", 5, 25, 10)
        
        if st.button("Generate Personalised Recommendations", type="primary"):
            with st.spinner("Scoring candidate questions across 6 hybrid signals..."):
                recs = engine.recommend_questions(target_company, solved_qids, top_k=top_k, model_type='hybrid')
                
                st.markdown(f"### Top Recommendations for **{target_company.title()}**")
                
                for idx, row in recs.iterrows():
                    topic_name = row.get('PrimaryTopic', row.get('primary_topic', 'General'))
                    
                    with st.expander(f"#{row['ID']} - {row['Title']} [{row['Difficulty']}] | Score: {row['RecommendationScore']}"):
                        st.markdown(f"**Primary Topic:** `{topic_name}`")
                        st.markdown(f"<div class='xai-badge'>{row['Explanation']}</div>", unsafe_allow_html=True)
                        
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Company Sim", f"{round(row['CompanySim']*100)}%")
                        m2.metric("Freq Bonus", f"{round(row['FreqBonus']*100)}%")
                        m3.metric("Topic Gap", f"+{round(row['TopicGap']*100)}%")
                        m4.metric("Diff Fit", f"{round(row['DiffFit']*100)}%")
                        m5.metric("Novelty", f"{round(row['Novelty']*100)}%")

    # MODE B: 8-WEEK PROGRESSIVE MMR ROADMAP
    else:
        rm_c1, rm_c2 = st.columns(2)
        with rm_c1:
            num_weeks = st.slider("Duration (Weeks):", 4, 12, 8)
        with rm_c2:
            q_per_week = st.slider("Questions per Week:", 2, 6, 4)
            
        if st.button("Generate Dynamic Study Roadmap", type="primary"):
            with st.spinner("Optimizing MMR roadmap schedule (<50ms fast vector execution)..."):
                roadmap = engine.generate_mmr_roadmap(target_company, solved_qids, num_weeks=num_weeks, q_per_week=q_per_week)
                
                st.markdown(f"### 8-Week Progressive Study Schedule for **{target_company.title()}**")
                st.dataframe(roadmap[['Week', 'TargetDiff', 'ID', 'Title', 'Difficulty', 'Topic', 'FinalMMRScore']], use_container_width=True)
                
                ch1, ch2 = st.columns(2)
                with ch1:
                    st.markdown("#### Weekly Difficulty Progression Curve")
                    fig_diff = px.line(roadmap.groupby('Week')['TargetDiff'].mean().reset_index(), x='Week', y='TargetDiff', markers=True, title="Target Difficulty Ramp (Sigmoid Curve)")
                    st.plotly_chart(fig_diff, use_container_width=True)
                with ch2:
                    st.markdown("#### Weekly Topic Breakdown")
                    fig_top = px.bar(roadmap, x='Week', color='Topic', title='Topic Diversity per Week (MMR Constraint)')
                    st.plotly_chart(fig_top, use_container_width=True)
                    
                csv_data = roadmap.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Study Roadmap (CSV)", data=csv_data, file_name=f"{target_company}_study_roadmap.csv", mime="text/csv")
