import torch
import os
import pandas as pd
import random
import numpy as np

class SetDset(torch.utils.data.Dataset):
    def __init__(self,csv,num_fram_per_set):
        super(SetDset,self).__init__()
        self.data_csv = pd.read_csv(csv)
        self.num_frame_per_set = num_fram_per_set

    def __getitem__(self, item):

        data = torch.load(self.data_csv['path'][item]).detach().cpu()
        target = self.data_csv['id'][item]
        subject_data = list(self.data_csv[self.data_csv['id']==target].index.values)
        subject_data.remove(item)
        selection = random.choices(subject_data,k=self.num_frame_per_set-1)
        for s in selection:
            temp = torch.load(self.data_csv['path'][s]).detach().cpu()
            data = torch.cat((data,temp),dim=0)

        return data,target

    def __len__(self):
        return len(self.data_csv)
    def num_class(self):
        return max(self.data_csv['id'])+1