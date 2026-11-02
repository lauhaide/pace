import torch
from torch import nn
from reranker.dataset_collection import SESingleDataset, PosNegSingleDataset, SEPairwiseDataset
#from swag import utils
from reranker.utils import predict
#from transformers import get_linear_schedule_with_warmup, AdamW
import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange
from torch import optim
import os


class VanillaBert(nn.Module):
    def __init__(self, base_model, pretrained_model, device) -> None:
        super().__init__()
        self.base_model = base_model
        self.base_model.to(device)
        self.pretrained_model = pretrained_model
        self.device = device

    def predict(self, test_data, question, samples_nums, eval):
        # dataset = SESingleDataset(log_data=test_data, question=question, pretrained_model=self.pretrained_model)
        dataset = PosNegSingleDataset(
            log_data=test_data, question=question, pretrained_model=self.pretrained_model)
        data_loader = DataLoader(dataset, batch_size=512, shuffle=False)

        self.base_model.train()
        if eval:
            samples_nums = 1

        all_scores = np.zeros((len(dataset), samples_nums))

        for i in trange(samples_nums):
            scores = predict(data_loader, self.base_model, eval)
            all_scores[:, i] = scores
            
        self.mean, self.var = np.mean(
            all_scores, axis=1), np.var(all_scores, axis=1)

        return self.mean, self.var

    def get_utilities(self, test_data, question, sample_nums=1):
        mean, _ = self.predict(test_data, question, sample_nums, eval=True)
        return mean