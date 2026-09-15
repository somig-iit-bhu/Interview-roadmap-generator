"""
Interview Intelligence Engine: High-Performance Recommendation & Roadmap Optimization Engine
Author: Placement Candidate (2026)
"""

import pandas as pd
import numpy as np
import ast
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class InterviewIntelligenceEngine:
    def __init__(self, data_dir='data/processed', embedding_path='question_embeddings.npy', hf_dataset_url=None):
        print("Initializing High-Performance Interview Intelligence Engine...")
        self.questions = pd.read_csv(f"{data_dir}/question_info.csv")
        
        # Smart Fail-Safe Loader for final_dataset.csv (Supports HF URL & Lightweight Fallback)
        self.final_ds = self._load_final_dataset(data_dir, hf_dataset_url)
        self.company_features = pd.read_csv(f"{data_dir}/all_company_feature_final.csv")
        self.embeddings = np.load(embedding_path)
        
        # Clean ID mapping
        self.questions['ID'] = self.questions['ID'].astype(int)
        self.final_ds['ID'] = self.final_ds['ID'].astype(int)
        
        # Build ID to Embedding Index lookup & reverse lookup
        self.id_to_idx = {qid: idx for idx, qid in enumerate(self.questions['ID'])}
        self.idx_to_id = {idx: qid for idx, qid in enumerate(self.questions['ID'])}

        # Parse topics into clean python list
        self.questions['parsed_topics'] = self.questions['topics'].apply(self._parse_topics)
        self.questions['PrimaryTopic'] = self.questions['parsed_topics'].apply(lambda x: x[0] if x else 'General')
        self.questions['primary_topic'] = self.questions['PrimaryTopic'] # Backwards compatibility
        
        # Difficulty Map to numeric score
        diff_map = {'Easy': 1.0, 'Medium': 2.0, 'Hard': 3.0}
        self.questions['difficulty_score'] = self.questions['Difficulty'].map(diff_map).fillna(2.0)
        
        # Build Question Text Corpus for TF-IDF Baseline
        self.questions['text_corpus'] = (
            self.questions['Title'].fillna('') + ' ' +
            self.questions['topics'].fillna('') + ' ' +
            self.questions['category'].fillna('') + ' ' +
            self.questions['description'].fillna('')
        )
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.questions['text_corpus'])
        
        # Build Company-Question Frequency Lookup: (company_name, qid) -> freq_pct
        self.company_q_freq = self._build_company_q_freq_lookup()
        
        # Build Global Company Embeddings (Frequency-Weighted)
        self.company_embeddings = self._build_company_embeddings_weighted(self.final_ds)
        print("Engine Initialized Successfully!")

    def _load_final_dataset(self, data_dir, hf_url=None):
        local_path = f"{data_dir}/final_dataset.csv"
        if os.path.exists(local_path):
            print(f"Loading local dataset: {local_path}")
            return pd.read_csv(local_path)
        
        # Check Hugging Face URL parameter or environment variable
        target_hf_url = hf_url or os.environ.get("HF_DATASET_URL")
        if target_hf_url:
            try:
                print(f"Downloading final_dataset.csv from Hugging Face URL: {target_hf_url}")
                df = pd.read_csv(target_hf_url)
                return df
            except Exception as e:
                print(f"Warning: Failed to load from Hugging Face URL ({e}).")

        # Graceful Fallback to company_question.csv (pushed to GitHub)
        fallback_path = f"{data_dir}/company_question.csv"
        if os.path.exists(fallback_path):
            print(f"Notice: {local_path} not found. Using GitHub fallback: {fallback_path}")
            df = pd.read_csv(fallback_path)
            if 'Frequency %' not in df.columns:
                df['Frequency %'] = 100.0
            return df

        raise FileNotFoundError(f"Neither {local_path} nor fallback dataset ({fallback_path}) could be loaded.")

    def _parse_topics(self, topic_str):
        if pd.isna(topic_str) or not topic_str:
            return ['General']
        if isinstance(topic_str, list):
            return topic_str
        try:
            val = ast.literal_eval(topic_str)
            if isinstance(val, list):
                return val
        except:
            pass
        return [t.strip() for t in str(topic_str).replace("'", "").replace("[", "").replace("]", "").split(",") if t.strip()]

    def _build_company_q_freq_lookup(self, df=None):
        if df is None:
            df = self.final_ds
        lookup = {}
        for _, row in df.iterrows():
            comp = str(row['company']).lower().strip()
            qid = int(row['ID'])
            freq = float(row.get('Frequency %', 100.0))
            lookup[(comp, qid)] = freq
        return lookup

    def _build_company_embeddings_weighted(self, df):
        company_vecs = {}
        for comp, group in df.groupby('company'):
            comp_name = str(comp).lower().strip()
            vecs = []
            weights = []
            for _, row in group.iterrows():
                qid = int(row['ID'])
                if qid in self.id_to_idx:
                    vecs.append(self.embeddings[self.id_to_idx[qid]])
                    weights.append(float(row.get('Frequency %', 1.0)))
            
            if vecs:
                vecs = np.array(vecs)
                weights = np.array(weights)
                w_sum = np.sum(weights)
                if w_sum > 0:
                    company_vecs[comp_name] = np.average(vecs, axis=0, weights=weights)
                else:
                    company_vecs[comp_name] = np.mean(vecs, axis=0)
                    
        return company_vecs

    def compute_user_profile(self, solved_qids):
        valid_indices = [self.id_to_idx[qid] for qid in solved_qids if qid in self.id_to_idx]
        if not valid_indices:
            user_vec = np.zeros(self.embeddings.shape[1])
            user_topic_dist = {}
        else:
            user_vec = np.mean(self.embeddings[valid_indices], axis=0)
            solved_df = self.questions[self.questions['ID'].isin(solved_qids)]
            all_topics = [t for topics in solved_df['parsed_topics'] for t in topics]
            total = max(len(all_topics), 1)
            user_topic_dist = {t: count / total for t, count in pd.Series(all_topics).value_counts().items()}
        return user_vec, user_topic_dist

    def compute_company_topic_profile(self, company_name, company_df=None):
        comp = company_name.lower().strip()
        if company_df is not None:
            comp_qids = company_df['ID'].tolist()
        else:
            comp_qids = [qid for (c, qid) in self.company_q_freq.keys() if c == comp]
            
        comp_df = self.questions[self.questions['ID'].isin(comp_qids)]
        all_topics = [t for topics in comp_df['parsed_topics'] for t in topics]
        total = max(len(all_topics), 1)
        return {t: count / total for t, count in pd.Series(all_topics).value_counts().items()}

    def recommend_questions(
        self, 
        target_company, 
        solved_qids, 
        top_k=20, 
        target_diff=2.0, 
        w_sim=0.35, 
        w_freq=0.20, 
        w_gap=0.20, 
        w_diff=0.15, 
        w_nov=0.10,
        model_type='hybrid',
        train_company_df=None
    ):
        comp = target_company.lower().strip()
        
        if train_company_df is not None:
            comp_embeddings = self._build_company_embeddings_weighted(train_company_df)
            comp_vec = comp_embeddings.get(comp, np.mean(self.embeddings, axis=0))
            comp_topics = self.compute_company_topic_profile(target_company, train_company_df)
            comp_q_freq_local = self._build_company_q_freq_lookup(train_company_df)
        else:
            comp_vec = self.company_embeddings.get(comp, np.mean(self.embeddings, axis=0))
            comp_topics = self.compute_company_topic_profile(target_company)
            comp_q_freq_local = self.company_q_freq

        user_vec, user_topics = self.compute_user_profile(solved_qids)
        solved_set = set(solved_qids)
        candidate_ids = [qid for qid in self.questions['ID'] if qid not in solved_set and qid in self.id_to_idx]
        candidates = self.questions[self.questions['ID'].isin(candidate_ids)].copy()
        
        # MODEL 1: Random Baseline
        if model_type == 'random':
            sampled = candidates.sample(n=min(top_k, len(candidates)), random_state=42).copy()
            sampled['RecommendationScore'] = np.random.uniform(0.1, 0.9, size=len(sampled))
            sampled['Explanation'] = "Random Selection Baseline"
            return sampled[['ID', 'Title', 'Difficulty', 'PrimaryTopic', 'RecommendationScore', 'Explanation']]

        # MODEL 2: Pure Frequency Baseline
        if model_type == 'frequency':
            results = []
            for _, row in candidates.iterrows():
                qid = row['ID']
                freq_pct = comp_q_freq_local.get((comp, qid), 0.0)
                results.append({
                    'ID': qid,
                    'Title': row['Title'],
                    'Difficulty': row['Difficulty'],
                    'PrimaryTopic': row['PrimaryTopic'],
                    'RecommendationScore': round(freq_pct / 100.0, 4),
                    'Explanation': f"Pure Frequency Baseline ({freq_pct}%)"
                })
            return pd.DataFrame(results).sort_values('RecommendationScore', ascending=False).head(top_k)

        # MODEL 3: TF-IDF Similarity Baseline
        if model_type == 'tfidf':
            if train_company_df is not None:
                train_qids = train_company_df['ID'].tolist()
            else:
                train_qids = [qid for (c, qid) in comp_q_freq_local.keys() if c == comp]
            
            train_indices = [self.id_to_idx[q] for q in train_qids if q in self.id_to_idx]
            if train_indices:
                company_tfidf_vec = self.tfidf_matrix[train_indices].mean(axis=0)
                company_tfidf_vec = np.asarray(company_tfidf_vec)
            else:
                company_tfidf_vec = np.zeros((1, self.tfidf_matrix.shape[1]))
                
            results = []
            for _, row in candidates.iterrows():
                qid = row['ID']
                idx = self.id_to_idx[qid]
                q_tfidf = self.tfidf_matrix[idx]
                sim = float(cosine_similarity(q_tfidf, company_tfidf_vec)[0][0])
                results.append({
                    'ID': qid,
                    'Title': row['Title'],
                    'Difficulty': row['Difficulty'],
                    'PrimaryTopic': row['PrimaryTopic'],
                    'RecommendationScore': round(sim, 4),
                    'Explanation': f"TF-IDF Similarity ({round(sim*100)}%)"
                })
            return pd.DataFrame(results).sort_values('RecommendationScore', ascending=False).head(top_k)

        # MODEL 4: Pure Sentence Transformer Embedding Similarity Baseline
        if model_type == 'sentence_transformer':
            results = []
            for _, row in candidates.iterrows():
                qid = row['ID']
                idx = self.id_to_idx[qid]
                q_vec = self.embeddings[idx]
                sim = float(cosine_similarity([q_vec], [comp_vec])[0][0])
                results.append({
                    'ID': qid,
                    'Title': row['Title'],
                    'Difficulty': row['Difficulty'],
                    'PrimaryTopic': row['PrimaryTopic'],
                    'RecommendationScore': round(sim, 4),
                    'Explanation': f"Sentence Transformer Embedding Sim ({round(sim*100)}%)"
                })
            return pd.DataFrame(results).sort_values('RecommendationScore', ascending=False).head(top_k)

        # MODEL 5: Full Hybrid Recommender (Ours)
        has_user_history = np.linalg.norm(user_vec) > 0
        results = []
        for _, row in candidates.iterrows():
            qid = row['ID']
            idx = self.id_to_idx[qid]
            q_vec = self.embeddings[idx]
            
            sim = float(cosine_similarity([q_vec], [comp_vec])[0][0])
            freq_pct = comp_q_freq_local.get((comp, qid), 0.0)
            freq_bonus = freq_pct / 100.0
            
            topics = row['parsed_topics']
            gaps = [max(0.0, comp_topics.get(t, 0.0) - user_topics.get(t, 0.0)) for t in topics]
            gap_score = float(np.mean(gaps)) if gaps else 0.0
            
            diff_score = float(1.0 - abs(row['difficulty_score'] - target_diff) / 2.0)
            
            if has_user_history:
                user_sim = float(cosine_similarity([q_vec], [user_vec])[0][0])
                novelty_score = float(1.0 - user_sim)
            else:
                novelty_score = 1.0
                
            final_score = (
                w_sim * sim + 
                w_freq * freq_bonus + 
                w_gap * gap_score + 
                w_diff * diff_score + 
                w_nov * novelty_score
            )
            
            explain_str = (
                f"CompanySim: {round(sim*100)}% | FreqBonus: {round(freq_bonus*100)}% | "
                f"TopicGap: +{round(gap_score*100)}% | DiffFit: {round(diff_score*100)}% | Novelty: {round(novelty_score*100)}%"
            )
            
            results.append({
                'ID': qid,
                'Title': row['Title'],
                'Difficulty': row['Difficulty'],
                'PrimaryTopic': row['PrimaryTopic'],
                'CompanySim': round(sim, 4),
                'FreqBonus': round(freq_bonus, 4),
                'TopicGap': round(gap_score, 4),
                'DiffFit': round(diff_score, 4),
                'Novelty': round(novelty_score, 4),
                'RecommendationScore': round(final_score, 4),
                'Explanation': explain_str
            })
            
        return pd.DataFrame(results).sort_values('RecommendationScore', ascending=False).head(top_k)

    def generate_mmr_roadmap(
        self, 
        target_company, 
        solved_qids, 
        num_weeks=8, 
        q_per_week=4, 
        lambda_param=0.7
    ):
        scheduled_qids = set(solved_qids)
        roadmap_rows = []

        candidates = self.recommend_questions(
            target_company=target_company,
            solved_qids=solved_qids,
            top_k=150,
            model_type='hybrid'
        )
        
        candidate_ids = candidates['ID'].tolist()
        candidate_indices = [self.id_to_idx[q] for q in candidate_ids if q in self.id_to_idx]
        candidate_vecs = self.embeddings[candidate_indices]
        
        cand_sim_matrix = cosine_similarity(candidate_vecs, candidate_vecs)
        
        for week in range(1, num_weeks + 1):
            target_diff = 1.0 + 2.0 / (1.0 + np.exp(-0.8 * (week - num_weeks / 2.0)))
            
            for _ in range(q_per_week):
                best_mmr = -float('inf')
                best_row = None
                best_redundancy = 0.0
                
                for idx_c, row in candidates.iterrows():
                    qid = row['ID']
                    if qid in scheduled_qids:
                        continue
                        
                    diff_score = 1.0 - abs(row['difficulty_score'] if 'difficulty_score' in row else 2.0 - target_diff) / 2.0
                    base_rel = row['RecommendationScore'] + 0.1 * diff_score
                    
                    if len(scheduled_qids) > len(solved_qids):
                        sch_cand_indices = [i for i, c_id in enumerate(candidate_ids) if c_id in scheduled_qids]
                        if sch_cand_indices:
                            cand_pos = candidate_ids.index(qid)
                            max_sim = float(np.max(cand_sim_matrix[cand_pos, sch_cand_indices]))
                        else:
                            max_sim = 0.0
                    else:
                        max_sim = 0.0
                        
                    mmr_score = lambda_param * base_rel - (1.0 - lambda_param) * max_sim
                    
                    if mmr_score > best_mmr:
                        best_mmr = mmr_score
                        best_row = row
                        best_redundancy = max_sim
                        
                if best_row is not None:
                    qid = best_row['ID']
                    scheduled_qids.add(qid)
                    
                    xai = f"{best_row['Explanation']} | MMR_Redundancy: -{round(best_redundancy*100)}%"
                    
                    roadmap_rows.append({
                        'Week': week,
                        'TargetDiff': round(target_diff, 2),
                        'ID': qid,
                        'Title': best_row['Title'],
                        'Topic': best_row['PrimaryTopic'],
                        'Difficulty': best_row['Difficulty'],
                        'FinalMMRScore': round(best_mmr, 4),
                        'Explanation': xai
                    })
                    
        return pd.DataFrame(roadmap_rows)

    def evaluate_leakage_free_experiment(self, companies=['google', 'amazon', 'meta', 'microsoft', 'apple', 'uber', 'adobe'], k=10):
        models = ['random', 'frequency', 'tfidf', 'sentence_transformer', 'hybrid']
        results_by_model = {m: {'Recall': [], 'Precision': [], 'NDCG': [], 'MAP': []} for m in models}

        for comp in companies:
            comp_clean = comp.lower().strip()
            comp_rows = self.final_ds[self.final_ds['company'].str.lower() == comp_clean].copy()
            if len(comp_rows) < 15:
                continue
                
            shuffled = comp_rows.sample(frac=1.0, random_state=42).reset_index(drop=True)
            n = len(shuffled)
            n_train = int(0.70 * n)
            n_val = int(0.15 * n)
            
            train_df = shuffled.iloc[:n_train]
            val_df = shuffled.iloc[n_train:n_train+n_val]
            test_df = shuffled.iloc[n_train+n_val:]
            
            test_qids = set(test_df['ID'].astype(int))
            train_qids = train_df['ID'].astype(int).tolist()
            
            for m in models:
                recs = self.recommend_questions(
                    target_company=comp, solved_qids=train_qids, top_k=k, model_type=m,
                    train_company_df=train_df
                )
                rec_ids = recs['ID'].tolist()
                
                hits = set(rec_ids).intersection(test_qids)
                recall = len(hits) / len(test_qids) if test_qids else 0.0
                precision = len(hits) / k
                
                dcg = sum([1.0 / np.log2(i + 2) for i, qid in enumerate(rec_ids) if qid in test_qids])
                idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(test_qids), k))])
                ndcg = dcg / idcg if idcg > 0 else 0.0
                
                ap = 0.0
                num_hits = 0
                for i, qid in enumerate(rec_ids):
                    if qid in test_qids:
                        num_hits += 1
                        ap += num_hits / (i + 1)
                map_k = ap / min(len(test_qids), k) if test_qids else 0.0
                
                results_by_model[m]['Recall'].append(recall)
                results_by_model[m]['Precision'].append(precision)
                results_by_model[m]['NDCG'].append(ndcg)
                results_by_model[m]['MAP'].append(map_k)

        table_rows = []
        model_names = {
            'random': 'Baseline 1: Random Search',
            'frequency': 'Baseline 2: Pure Company Frequency',
            'tfidf': 'Baseline 3: TF-IDF Lexical Similarity',
            'sentence_transformer': 'Baseline 4: Sentence Transformer (MiniLM)',
            'hybrid': 'Full Hybrid Recommender (Ours)'
        }
        
        for m in models:
            table_rows.append({
                'Model Architecture': model_names[m],
                f'Recall@{k}': round(float(np.mean(results_by_model[m]['Recall'])), 4),
                f'Precision@{k}': round(float(np.mean(results_by_model[m]['Precision'])), 4),
                f'NDCG@{k}': round(float(np.mean(results_by_model[m]['NDCG'])), 4),
                f'MAP@{k}': round(float(np.mean(results_by_model[m]['MAP'])), 4)
            })
            
        df_final = pd.DataFrame(table_rows)
        df_final.to_csv('data/processed/model_benchmark_results.csv', index=False)
        return df_final

if __name__ == '__main__':
    engine = InterviewIntelligenceEngine()
    print("Testing recommendations...")
    recs = engine.recommend_questions('google', [1, 2], top_k=3)
    print(recs[['ID', 'Title', 'RecommendationScore']])
