import os
import torch
from torch.utils.data import DataLoader
from utils import *
from Models.NeuralAggregation import Aggregation
from Dataset.SetDataset import SetDset
import os
import config

cfg = config.cfg()

if not os.path.exists(cfg.log_path):
    os.mkdir(cfg.log_path)

init_logging(cfg.device, cfg.log_path)

print('device is: ',cfg.device)

logging.info(cfg.des)

dataset = SetDset(csv = cfg.csv_dir,num_fram_per_set=5)
num_classes= dataset.num_class()
train_loader = DataLoader(dataset= dataset,batch_size=cfg.batch_size,shuffle=True)

# create model
model = Aggregation(num_features=512,num_classes=num_classes,num_hidden = 256)
model = model.to(cfg.device)

# create optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr,momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

print('start training ... ')
for epoch in range(cfg.max_epoch):
    tr_acc, tr_loss = trainNAN(epoch,model,train_loader,optimizer,criterion,cfg.device)
    logging.info(f'train_acc= {tr_acc}, train_loss= {tr_loss}, epoch= {epoch}')
    scheduler.step()

    print('Saving...')
    state = {
            'net': model.state_dict(),
            'acc': tr_acc,
            'epoch': epoch,
            }
    torch.save(state, os.path.join(cfg.log_path,f'NAN_{epoch}.pth'))
