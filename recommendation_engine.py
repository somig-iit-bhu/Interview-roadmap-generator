"""
Interview Intelligence Engine: Production-Grade Multi-Objective Recommendation,
Personalization, Roadmap & Evaluation System
Author: IIT BHU Candidate (Placement 2026)
"""

import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity

class InterviewIntelligenceEngine:
    def __init__(self, data_dir='data/processed', embedding_path='question_embeddings.npy'):
        print("Initializing Interview Intelligence Engine v2.0...")
        self.questions = pd.read_csv(f"{data_dir}/question_info.csv")
        self.final_ds = pd.read_csv(f"{data_dir}/final_dataset.csv")
        self.company_features = pd.read_csv(f"{data_dir}/all_company_feature_final.csv")
        self.embeddings = np.load(embedding_path)
        
        # Clean ID mapping
        self.questions['ID'] = self.questions['ID'].astype(int)
        self.final_ds['ID'] = self.final_ds['ID'].astype(int)
        
        # Build ID to Embedding Index lookup
        self.id_to_idx = {qid: idx for idx, qid in enumerate(self.questions['ID'])}
        
        # Parse topics into clean python list
        self.questions['parsed_topics'] = self.questions['topics'].apply(self._parse_topics)
        
        # Difficulty Map to numeric score
        diff_map = {'Easy': 1.0, 'Medium': 2.0, 'Hard': 3.0}
        self.questions['difficulty_score'] = self.questions['Difficulty'].map(diff_map).fillna(2.0)
        
        # Build Company-Question Frequency Lookup: (company_name, qid) -> freq_pct
        self.company_q_freq = self._build_company_q_freq_lookup()
        
        # Build Frequency-Weighted Company Embeddings (Improvement #1)
        self.company_embeddings = self._build_company_embeddings_weighted()
        print("Engine initialized successfully with 10/10 technical upgrades!")

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

    def _build_company_q_freq_lookup(self):
        lookup = {}
        for _, row in self.final_ds.iterrows():
            comp = str(row['company']).lower().strip()
            qid = int(row['ID'])
            freq = float(row.get('Frequency %', 100.0))
            lookup[(comp, qid)] = freq
        return lookup

    def _build_company_embeddings_weighted(self):
        """Improvement #1: Frequency-Weighted Company Embeddings: e_C = sum(f_i * e_i) / sum(f_i)"""
        company_vecs = {}
        for comp, group in self.final_ds.groupby('company'):
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
        """Improvement #5: Computes User Vector (e_U) and Multi-Topic Distribution."""
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

    def compute_company_topic_profile(self, company_name):
        comp = company_name.lower().strip()
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
        w_nov=0.10
    ):
        """
        Improvement #2 & #5 & #6 & #7 & #8:
        Scores ALL 3,500 questions with 6-Factor Hybrid Formula + XAI Explainability Breakdown.
        """
        comp = target_company.lower().strip()
        comp_vec = self.company_embeddings.get(comp, np.mean(self.embeddings, axis=0))
        user_vec, user_topics = self.compute_user_profile(solved_qids)
        comp_topics = self.compute_company_topic_profile(target_company)
        
        # Candidate selection: ALL unsolved questions (Improvement #2)
        solved_set = set(solved_qids)
        candidate_ids = [qid for qid in self.questions['ID'] if qid not in solved_set and qid in self.id_to_idx]
        candidates = self.questions[self.questions['ID'].isin(candidate_ids)].copy()
        
        has_user_history = np.linalg.norm(user_vec) > 0
        
        results = []
        for _, row in candidates.iterrows():
            qid = row['ID']
            idx = self.id_to_idx[qid]
            q_vec = self.embeddings[idx]
            
            # 1. Company Similarity (Cosine)
            sim = float(cosine_similarity([q_vec], [comp_vec])[0][0])
            
            # 2. Frequency Bonus (0 to 1) - Improvement #2
            freq_pct = self.company_q_freq.get((comp, qid), 0.0)
            freq_bonus = freq_pct / 100.0
            
            # 3. Multi-Tag Topic Deficit Gap - Improvement #6
            topics = row['parsed_topics']
            gaps = [max(0.0, comp_topics.get(t, 0.0) - user_topics.get(t, 0.0)) for t in topics]
            gap_score = float(np.mean(gaps)) if gaps else 0.0
            
            # 4. Target Difficulty Alignment - Improvement #3
            diff_score = float(1.0 - abs(row['difficulty_score'] - target_diff) / 2.0)
            
            # 5. User Personalization Novelty Score - Improvement #5
            if has_user_history:
                user_sim = float(cosine_similarity([q_vec], [user_vec])[0][0])
                novelty_score = float(1.0 - user_sim) # Penalize identical questions to user history
            else:
                novelty_score = 1.0
                
            # Hybrid Recommendation Score
            final_score = (
                w_sim * sim + 
                w_freq * freq_bonus + 
                w_gap * gap_score + 
                w_diff * diff_score + 
                w_nov * novelty_score
            )
            
            # Improvement #8: Explainability Breakdown (XAI)
            explain_str = (
                f"CompanySim: {round(sim*100)}% | FreqBonus: {round(freq_bonus*100)}% | "
                f"TopicGap: +{round(gap_score*100)}% | DiffFit: {round(diff_score*100)}% | Novelty: {round(novelty_score*100)}%"
            )
            
            results.append({
                'ID': qid,
                'Title': row['Title'],
                'Difficulty': row['Difficulty'],
                'PrimaryTopic': topics[0] if topics else 'General',
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
        """
        Improvement #3 & #4:
        Dynamic Difficulty Progression (TargetDiff ramps Easy -> Hard) + Recomputed Utility MMR.
        """
        scheduled_qids = set(solved_qids)
        roadmap_rows = []

        for week in range(1, num_weeks + 1):
            # Improvement #3: Smooth Sigmoid Difficulty Ramp (Easy 1.0 -> Medium 2.0 -> Hard 2.8)
            target_diff = 1.0 + 2.0 / (1.0 + np.exp(-0.8 * (week - num_weeks / 2.0)))
            
            for _ in range(q_per_week):
                # Improvement #4: Dynamic Utility Re-scoring at every single iteration
                candidates = self.recommend_questions(
                    target_company=target_company,
                    solved_qids=list(scheduled_qids),
                    top_k=50,
                    target_diff=target_diff
                )
                
                best_mmr = -float('inf')
                best_row = None
                best_redundancy = 0.0
                
                for _, row in candidates.iterrows():
                    qid = row['ID']
                    if qid in scheduled_qids:
                        continue
                        
                    idx = self.id_to_idx[qid]
                    q_vec = self.embeddings[idx]
                    base_rel = row['RecommendationScore']
                    
                    # MMR Redundancy Penalty against questions scheduled in roadmap so far
                    if len(scheduled_qids) > len(solved_qids):
                        scheduled_vecs = [self.embeddings[self.id_to_idx[s]] for s in scheduled_qids if s in self.id_to_idx]
                        max_sim = float(np.max(cosine_similarity([q_vec], scheduled_vecs)[0]))
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

    def evaluate_offline(self, target_company='google', k=10):
        """Improvement #9: Multi-Company Evaluation Module (Recall@K, Precision@K, NDCG@K)."""
        comp = target_company.lower().strip()
        comp_qids = list(set([qid for (c, qid) in self.company_q_freq.keys() if c == comp]))
        
        if len(comp_qids) < 10:
            return None
            
        np.random.seed(42)
        shuffled = np.random.permutation(comp_qids)
        split_idx = int(0.8 * len(shuffled))
        train_qids, test_qids = set(shuffled[:split_idx]), set(shuffled[split_idx:])
        
        recs = self.recommend_questions(target_company, solved_qids=list(train_qids), top_k=k)
        rec_ids = list(recs['ID'])
        
        hits = set(rec_ids).intersection(test_qids)
        recall_at_k = len(hits) / len(test_qids) if test_qids else 0.0
        precision_at_k = len(hits) / k
        
        dcg = sum([1.0 / np.log2(i + 2) for i, qid in enumerate(rec_ids) if qid in test_qids])
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(test_qids), k))])
        ndcg_at_k = dcg / idcg if idcg > 0 else 0.0
        
        return {
            'Company': target_company,
            'Test_Questions': len(test_qids),
            f'Recall@{k}': round(recall_at_k, 4),
            f'Precision@{k}': round(precision_at_k, 4),
            f'NDCG@{k}': round(ndcg_at_k, 4)
        }

    def evaluate_multi_company_benchmark(self, companies=['google', 'amazon', 'meta', 'microsoft', 'apple', 'uber', 'adobe'], k=10):
        """Improvement #9: Evaluates average metrics across major tech companies."""
        results = []
        for comp in companies:
            res = self.evaluate_offline(comp, k=k)
            if res:
                results.append(res)
        df_res = pd.DataFrame(results)
        
        avg_row = {
            'Company': 'AVERAGE_BENCHMARK',
            'Test_Questions': df_res['Test_Questions'].mean(),
            f'Recall@{k}': df_res[f'Recall@{k}'].mean(),
            f'Precision@{k}': df_res[f'Precision@{k}'].mean(),
            f'NDCG@{k}': df_res[f'NDCG@{k}'].mean()
        }
        return pd.concat([df_res, pd.DataFrame([avg_row])], ignore_index=True)

    def grid_search_weights(self, validation_companies=['google', 'amazon', 'meta'], k=10):
        """Improvement #10: Data-Driven Grid Search for Recommendation Scoring Weights."""
        candidate_weights = [
            (0.35, 0.20, 0.20, 0.15, 0.10),
            (0.40, 0.25, 0.15, 0.10, 0.10),
            (0.30, 0.30, 0.20, 0.10, 0.10),
            (0.50, 0.20, 0.10, 0.10, 0.10)
        ]
        
        best_ndcg = -1.0
        best_weights = None
        
        for w_tuple in candidate_weights:
            w_sim, w_freq, w_gap, w_diff, w_nov = w_tuple
            ndcg_list = []
            
            for comp in validation_companies:
                comp_qids = list(set([qid for (c, qid) in self.company_q_freq.keys() if c == comp.lower()]))
                if len(comp_qids) < 10:
                    continue
                np.random.seed(42)
                shuffled = np.random.permutation(comp_qids)
                train_qids, test_qids = set(shuffled[:int(0.8*len(shuffled))]), set(shuffled[int(0.8*len(shuffled)):])
                
                recs = self.recommend_questions(
                    comp, solved_qids=list(train_qids), top_k=k,
                    w_sim=w_sim, w_freq=w_freq, w_gap=w_gap, w_diff=w_diff, w_nov=w_nov
                )
                rec_ids = list(recs['ID'])
                
                dcg = sum([1.0 / np.log2(i + 2) for i, qid in enumerate(rec_ids) if qid in test_qids])
                idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(test_qids), k))])
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_list.append(ndcg)
                
            mean_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
            if mean_ndcg > best_ndcg:
                best_ndcg = mean_ndcg
                best_weights = w_tuple
                
        return {
            'Best_Weights_(w_sim, w_freq, w_gap, w_diff, w_nov)': best_weights,
            'Validation_Mean_NDCG@10': round(float(best_ndcg), 4)
        }

if __name__ == '__main__':
    engine = InterviewIntelligenceEngine()
    test_solved = [1, 2, 3] # Two Sum, etc.
    
    print("\n=======================================================")
    print(" 1. TOP 5 RECOMMENDATIONS (WITH EXPLAINABILITY XAI) ")
    print("=======================================================")
    recs = engine.recommend_questions('google', test_solved, top_k=5)
    for _, r in recs.iterrows():
        print(f"[{r['ID']}] {r['Title']} | Topic: {r['PrimaryTopic']} | Score: {r['RecommendationScore']}")
        print(f"    XAI: {r['Explanation']}\n")

    print("=======================================================")
    print(" 2. 4-WEEK ROADMAP (DYNAMIC DIFFICULTY + MMR) ")
    print("=======================================================")
    roadmap = engine.generate_mmr_roadmap('google', test_solved, num_weeks=4, q_per_week=2)
    print(roadmap[['Week', 'TargetDiff', 'ID', 'Title', 'Difficulty', 'Topic', 'FinalMMRScore']])

    print("\n=======================================================")
    print(" 3. MULTI-COMPANY OFFLINE BENCHMARK EVALUATION ")
    print("=======================================================")
    multi_eval = engine.evaluate_multi_company_benchmark(['google', 'amazon', 'meta', 'microsoft', 'apple'])
    print(multi_eval)

    print("\n=======================================================")
    print(" 4. DATA-DRIVEN GRID SEARCH WEIGHT OPTIMIZATION ")
    print("=======================================================")
    gs_res = engine.grid_search_weights(['google', 'amazon', 'meta'])
    print(gs_res)
