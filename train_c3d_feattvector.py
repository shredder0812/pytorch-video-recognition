import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from network.C3D_model_to_OSNET import C3D
from dataloaders.dataset_endocv import EndoscopyVideoDataset  # Đảm bảo file dataset nằm trong dataloaders/dataset.py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 50  # Số epoch huấn luyện
resume_epoch = 0  # Bắt đầu từ 0 hoặc tiếp tục
useTest = True  # Đánh giá trên tập test
nTestInterval = 10  # Đánh giá test mỗi 20 epoch
snapshot = 10  # Lưu model mỗi 50 epoch
lr = 1e-4  # Learning rate

dataset = 'endocv'  # Tên dataset
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'C3D'
saveName = modelName + '-' + dataset

def train_model(dataset=dataset, save_dir=save_dir, lr=lr, num_epochs=nEpochs, 
                save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    # Khởi tạo model
    model = C3D(pretrained=False)  # Dùng pretrained nếu có
    train_params = model.parameters()  # Huấn luyện tất cả tham số

    # Loss và optimizer
    criterion = nn.TripletMarginLoss(margin=0.5)
    optimizer = optim.Adam(train_params, lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True) # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if resume_epoch == 0:
        print(f"Training {modelName} from scratch...")
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                               map_location=lambda storage, loc: storage)
        print(f"Initializing weights from: {os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')}")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print(f'Training model on {dataset} dataset...')
    train_dataloader = DataLoader(EndoscopyVideoDataset(root_dir=f'endoc3d_data', split='train', clip_len=16), 
                                 batch_size=10, shuffle=True, num_workers=2)  # Giảm num_workers xuống 2
    val_dataloader = DataLoader(EndoscopyVideoDataset(root_dir=f'endoc3d_data', split='val', clip_len=16), 
                               batch_size=10, num_workers=2)  # Giảm num_workers xuống 2
    test_dataloader = DataLoader(EndoscopyVideoDataset(root_dir=f'endoc3d_data', split='test', clip_len=16), 
                                batch_size=10, num_workers=2)  # Giảm num_workers xuống 2

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    # Trong train_c3d_feattvector.py
    for epoch in range(resume_epoch, num_epochs):
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()
            running_loss = 0.0
            running_ap_dist = 0.0  # Khoảng cách anchor-positive
            running_an_dist = 0.0  # Khoảng cách anchor-negative

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for anchor, positive, negative in tqdm(trainval_loaders[phase]):
                anchor = Variable(anchor, requires_grad=True).to(device)
                positive = Variable(positive).to(device)
                negative = Variable(negative).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    anchor_feat = model(anchor)
                    positive_feat = model(positive)
                    negative_feat = model(negative)
                else:
                    with torch.no_grad():
                        anchor_feat = model(anchor)
                        positive_feat = model(positive)
                        negative_feat = model(negative)

                loss = criterion(anchor_feat, positive_feat, negative_feat)

                # Tính khoảng cách
                ap_dist = torch.nn.functional.pairwise_distance(anchor_feat, positive_feat).mean()
                an_dist = torch.nn.functional.pairwise_distance(anchor_feat, negative_feat).mean()

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * anchor.size(0)
                running_ap_dist += ap_dist.item() * anchor.size(0)
                running_an_dist += an_dist.item() * anchor.size(0)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_ap_dist = running_ap_dist / trainval_sizes[phase]
            epoch_an_dist = running_an_dist / trainval_sizes[phase]

            writer.add_scalar(f'data/{phase}_loss_epoch', epoch_loss, epoch)
            writer.add_scalar(f'data/{phase}_ap_dist', epoch_ap_dist, epoch)
            writer.add_scalar(f'data/{phase}_an_dist', epoch_an_dist, epoch)
            print(f"[{phase}] Epoch: {epoch+1}/{num_epochs} Loss: {epoch_loss}, AP Dist: {epoch_ap_dist}, AN Dist: {epoch_an_dist}")

            if phase == 'val':
                scheduler.step(epoch_loss)

            stop_time = timeit.default_timer()
            print(f"Execution time: {stop_time - start_time}\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print(f"Save model at {os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')}\n")

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()
            running_loss = 0.0

            for anchor, positive, negative in tqdm(test_dataloader):
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                with torch.no_grad():
                    anchor_feat = model(anchor)
                    positive_feat = model(positive)
                    negative_feat = model(negative)
                    loss = criterion(anchor_feat, positive_feat, negative_feat)

                running_loss += loss.item() * anchor.size(0)

            epoch_loss = running_loss / test_size
            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            print(f"[test] Epoch: {epoch+1}/{num_epochs} Loss: {epoch_loss}")
            stop_time = timeit.default_timer()
            print(f"Execution time: {stop_time - start_time}\n")

    writer.close()

if __name__ == "__main__":
    train_model()