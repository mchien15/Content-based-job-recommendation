import pandas as pd
import scipy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
from bert_extract_feature import extract_feature
import joblib
import os



def feature_extraction_job(final_df_jobs, type):
    assert type in range(4)

    if not os.path.exists('pkl_file'):
        os.mkdir('pkl_file')
    if not os.path.exists('npz_file'):
        os.mkdir('npz_file')
    if type == 0:
        """
        tf-idf vectorize and using cosine for compute similarity
        """
        if not os.path.exists('pkl_file/tfidf_vectorizer.pkl') or not os.path.exists('npz_file/tfidf_jobid.npz'):
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_jobid = tfidf_vectorizer.fit_transform((final_df_jobs['text'])) 

            assert tfidf_jobid.shape[0] == len(final_df_jobs)
            assert tfidf_jobid.shape[1] == len(tfidf_vectorizer.get_feature_names_out())

            scipy.sparse.save_npz('npz_file/tfidf_jobid.npz', tfidf_jobid)
            joblib.dump(tfidf_vectorizer, 'pkl_file/tfidf_vectorizer.pkl')
            return tfidf_jobid, tfidf_vectorizer
        else:
            tfidf_jobid = scipy.sparse.load_npz('npz_file/tfidf_jobid.npz')
            tfidf_vectorizer = joblib.load('pkl_file/tfidf_vectorizer.pkl')
            
            assert tfidf_jobid.shape[0] == len(final_df_jobs)
            assert tfidf_jobid.shape[1] == len(tfidf_vectorizer.get_feature_names_out())

            return tfidf_jobid, tfidf_vectorizer


    elif type == 1:
        """
        Bag-of-word vectorize and using cosine for compute similarity
        """
        if not os.path.exists('pkl_file/count_vectorizer.pkl') or not os.path.exists('npz_file/count_jobid.npz'):
            count_vectorizer = CountVectorizer()
            count_jobid = count_vectorizer.fit_transform((final_df_jobs['text'])) 

            assert count_jobid.shape[0] == len(final_df_jobs)
            assert count_jobid.shape[1] == len(count_vectorizer.get_feature_names_out())

            scipy.sparse.save_npz('npz_file/count_jobid.npz', count_jobid)
            joblib.dump(count_vectorizer, 'pkl_file/count_vectorizer.pkl')

            return count_jobid, count_vectorizer
        else:
            count_jobid = scipy.sparse.load_npz('npz_file/count_jobid.npz')  
            count_vectorizer = joblib.load('pkl_file/count_vectorizer.pkl')

            assert count_jobid.shape[0] == len(final_df_jobs)
            assert count_jobid.shape[1] == len(count_vectorizer.get_feature_names_out())

            return count_jobid, count_vectorizer
    
    elif type == 2:
        """
        tf-idf vectorize and using KNN for compute similarity
        """
        if not os.path.exists('pkl_file/knn.pkl') or not os.path.exists('npz_file/tfidf_jobid.npz'):
            tfidf_vectorizer = TfidfVectorizer()   
            tfidf_jobid = tfidf_vectorizer.fit_transform((final_df_jobs['text'])) 
            assert tfidf_jobid.shape[0] == len(final_df_jobs)
            assert tfidf_jobid.shape[1] == len(tfidf_vectorizer.get_feature_names_out())    

            n_neighbors = 10
            KNN = NearestNeighbors(n_neighbors=n_neighbors, p=2)
            KNN.fit(tfidf_jobid)

            scipy.sparse.save_npz('npz_file/tfidf_jobid.npz', tfidf_jobid)
            joblib.dump(KNN, 'pkl_file/knn.pkl')
            joblib.dump(tfidf_vectorizer, 'pkl_file/tfidf_vectorizer.pkl')

            return KNN, tfidf_jobid, tfidf_vectorizer
        else:
            tfidf_jobid = scipy.sparse.load_npz('npz_file/tfidf_jobid.npz')
            KNN = joblib.load('pkl_file/knn.pkl')        
            tfidf_vectorizer = joblib.load('pkl_file/tfidf_vectorizer.pkl')

            assert tfidf_jobid.shape[0] == len(final_df_jobs)
            assert tfidf_jobid.shape[1] == len(tfidf_vectorizer.get_feature_names_out())    

            return KNN, tfidf_jobid, tfidf_vectorizer

    elif type == 3:
        """
        pretrained BERT vectorize and using cosine for compute similarity
        """
        if not os.path.exists('./bert_pretrained_vector/pos_feature_vector.pt'):
            position_feature_vector = extract_feature()
        else:
            position_feature_vector = torch.load('./bert_pretrained_vector/pos_feature_vector.pt')
        return position_feature_vector


def feature_extraction_user(user_text, final_df_jobs, type):
    assert type in range(4)
    
    if type == 0:
        _, tfidf_vectorizer = feature_extraction_job(final_df_jobs, type)
        user_vector_tfidf = tfidf_vectorizer.transform(user_text)
        return user_vector_tfidf.toarray()

    elif type == 1:
        _, count_vectorizer = feature_extraction_job(final_df_jobs, type)
        user_vector_count = count_vectorizer.transform(user_text)
        return user_vector_count.toarray()

    elif type == 2:
        KNN, tfidf_jobid, tfidf_vectorizer = feature_extraction_job(final_df_jobs, type)
        user_vector_tfidf = tfidf_vectorizer.transform(user_text)
        NNS = KNN.kneighbors(user_vector_tfidf, return_distance=True)
        return NNS[1][0], NNS[0][0]
    
    elif type == 3:
        position_feature_vector = feature_extraction_job(final_df_jobs, type)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_basic_tokenization=True)
        model = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

        tokens = tokenizer([user_text], max_length=40, padding='max_length', truncation=True)
        user_feature_vector = model(torch.tensor(tokens['input_ids']), attention_mask=torch.tensor(tokens['attention_mask'])).hidden_states[-1]
        user_feature_vector = user_feature_vector.mean(axis=1)
        return position_feature_vector, user_feature_vector


