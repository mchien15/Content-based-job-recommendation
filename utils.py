import pandas as pd
import torch
import torch.nn as nn
import time
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from extract_feature import feature_extraction_job, feature_extraction_user
from clean import clean_txt


combine_df_job = pd.read_csv('./data/Combined_Jobs_Final.csv')
df_job = pd.read_csv('./data/Combined_Jobs_Final.csv')
df_job['Position'] = df_job['Position'].apply(clean_txt)
df_job = df_job[df_job['Position'] != '']

pos2id = df_job.groupby('Position')['Job.ID'].apply(list).to_dict()
job_position = list(pos2id.keys())


def compute_similarity(job_feature_vector, user_feature_vector):
    cos_similarity_tfidf = map(lambda x: cosine_similarity(user_feature_vector, x), job_feature_vector)
    return list(cos_similarity_tfidf)

def compute_similarity_bert(job_feature_vector, user_feature_vector):
    cos = nn.CosineSimilarity(dim=1)
    cosine_scores = cos(job_feature_vector, user_feature_vector)
    return cosine_scores

def get_applicant_id_info(applicant_id, df_job_view, df_experience, df_interest_position):
    try:
        position_view = ' và '.join(df_job_view[df_job_view['Applicant.ID'] == applicant_id]['Position'].values.tolist()).strip()
        company_view = ' và '.join(df_job_view[df_job_view['Applicant.ID'] == applicant_id]['Company'].values.tolist()).strip()
        city_view = ' và '.join(df_job_view[df_job_view['Applicant.ID'] == applicant_id]['City'].values.tolist()).strip()
    except:
        position_view = company_view = city_view = None
    
    try:
        position_experience = ' và '.join(df_experience[df_experience['Applicant.ID'] == applicant_id]['Position.Name'].values.tolist()).strip()
    except:
        position_experience = None

    try:
        position_interest = ' và '.join(df_interest_position[df_interest_position['Applicant.ID'] == applicant_id]['Position.Of.Interest'].values.tolist()).strip()
    except:
        position_interest = None

    print(f"Applicant Id: {applicant_id} đã bấm vào tin tức tuyển dụng trên website với \
            \nVị trí: {position_view} \
            \nTên công ty tương ứng là: {company_view} \
            \nThành phố tương ứng là: {city_view} \
            \nVị trí có kinh nghiệm: {position_experience} \
            \nVị trí mong muốn: {position_interest}")


def get_scores(feature_text, final_df_jobs, random_applicant_id, type):
    start_time = time.time()
    assert type in range(4)
    if type == 0 or type == 1:
        """
        Tf-idf and Bag-of-word
        """
        try:
            user_feature_vector = feature_extraction_user([feature_text], final_df_jobs, type)
            job_feature_vector, _ = feature_extraction_job(final_df_jobs, type)
            cosine_scores = compute_similarity(job_feature_vector, user_feature_vector)
            top = sorted(range(len(cosine_scores)), key=lambda i: cosine_scores[i], reverse=True)[:10]
            list_scores = [cosine_scores[i][0][0] for i in top]
            print(f"Time excution: {time.time() - start_time:.6f}s")
            return top, list_scores
        except:
            print(f"Applicant ID: {random_applicant_id} không để lại thông tin")
            
    elif type == 2:
        """
        KNN
        """
        top, list_scores = feature_extraction_user([feature_text], final_df_jobs, type)
        print(f"Time excution: {time.time() - start_time:.6f}s")
        return top, list_scores
    
    elif type == 3:
        """
        BERT
        """
        position_feature_vector, user_feature_vector = feature_extraction_user(feature_text, final_df_jobs, type)
        cosine_scores = compute_similarity_bert(position_feature_vector, user_feature_vector)
        scores, indices = torch.topk(cosine_scores, k=10)
        top = indices.tolist()
    
        top_idx2scores = {}
        for i, j in zip(scores, top):
            job_id = pos2id[job_position[j]]
            for idx in job_id:
                index = combine_df_job[combine_df_job['Job.ID'] == idx].index[0]
                top_idx2scores[index] = i.item()

        top = list(top_idx2scores.keys())
        list_scores = list(top_idx2scores.values())
        print(f"Time excution: {time.time() - start_time:.6f}s")
        return top, list_scores
    

def get_recommendation(top, df_job, scores, applicant_id=None):
    recommendation = pd.DataFrame()
    count = 0
    for i in top:
        if not applicant_id:
            recommendation.at[count, 'ApplicantID'] = applicant_id
        recommendation.at[count, 'JobID'] = int(df_job['Job.ID'][i])
        recommendation.at[count, 'title'] = df_job['Title'][i]
        recommendation.at[count, 'Position'] = df_job['Position'][i]
        recommendation.at[count, 'Company'] = df_job['Company'][i]
        recommendation.at[count, 'City'] = df_job['City'][i]
        recommendation.at[count, 'Job.Description'] = df_job['Job.Description'][i]
        recommendation.at[count, 'Employment.Type'] = df_job['Employment.Type'][i]
        recommendation.at[count, 'score'] =  scores[count]
        count += 1
    return recommendation