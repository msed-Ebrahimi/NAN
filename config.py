import torch
import os

class cfg():
    lr = 0.1
    best_acc = -1000.
    max_epoch = 100
    batch_size = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_path = 'experiments'
    csv_dir = 'sample_data.csv'
    des = f'CASIA face is used (MTCNN(face detection) + InceptionResnet(feature extraction))' \
          f'aggregation module, which takes 5 fram per person'