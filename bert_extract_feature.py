from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from clean import clean_txt
import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm



tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_basic_tokenization=True)
model = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

df_job = pd.read_csv('./data/Combined_Jobs_Final.csv')
df_job['Position'] = df_job['Position'].apply(clean_txt)
df_job = df_job[df_job['Position'] != '']

pos2id = df_job.groupby('Position')['Job.ID'].apply(list).to_dict()
job_position = list(pos2id.keys())


class PositionDataset(Dataset):
    def __init__(self, job_position):
        super().__init__()
        self.job_position = job_position

    def __len__(self):
        return len(self.job_position)

    def __getitem__(self, index):
        return self.job_position[index]


def data_collate_fn(batch_samples_list):
    arr = np.array(batch_samples_list)
    inputs = tokenizer(arr.tolist(), max_length=5, padding='max_length', truncation=True)
    return inputs


pos_dataset = PositionDataset(job_position)
pos_dataloader = DataLoader(pos_dataset, batch_size=64, collate_fn=data_collate_fn)

N = len(pos_dataloader.dataset)
T = 5
D = 768
i = 0

def extract_feature():
    with torch.no_grad():
        output = torch.zeros(N, T, D)
        for batch in tqdm(pos_dataloader):
            input = batch['input_ids']
            mask = batch['attention_mask']
            feature_vector = model(torch.tensor(input), attention_mask=torch.tensor(mask)).hidden_states[-1]

            if len(batch['input_ids']) == 64:
                output[i: i + 64] = feature_vector
            else:
                output[i: i + len(batch['input_ids'])] = feature_vector
            i += 64
    position_feature_vector = torch.mean(output, axis=1)
    torch.save(position_feature_vector, './bert_pretrained_vector/pos_feature_vector.pt')
    return position_feature_vector