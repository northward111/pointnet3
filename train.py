# https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a
from datetime import datetime
import numpy as np
import math
import random
import os
import torch
import pyperclip
from source import model
from source import dataset
from source import utils
from source.args import parse_args
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from config import CONFIG
from torch.utils.tensorboard import SummaryWriter

random.seed = 42
text_tag = 'main'
run_id = '{}{:02d}'.format(datetime.now().strftime('%y%m%d_%H%M%S_'), random.randint(0, 99))
log_path = CONFIG.bin_path / f'runs/torch_gan/{run_id}'
save_path = CONFIG.bin_path / 'checkpoints' / f'run_{run_id}'
log_path.mkdir(parents=True, exist_ok=True)
save_path.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_path)


def pointnetloss(outputs, labels, m3x3, m64x64, alpha=0.0001):
    criterion = torch.nn.NLLLoss()
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)


def train():
    path = CONFIG.dataset_path
    train_transforms = transforms.Compose([
        utils.PointSampler(1024),
        utils.Normalize(),
        utils.RandRotation_z(),
        utils.RandomNoise(),
        utils.ToTensor()
    ])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer.add_text(text_tag, f'device {device}')
    pointnet = model.PointNet()
    pointnet.to(device)
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=CONFIG.lr)

    train_ds = dataset.PointCloudData(path, transform=train_transforms)
    valid_ds = dataset.PointCloudData(path, valid=True, folder='test', transform=train_transforms)
    writer.add_text(text_tag, f'Train dataset size: {len(train_ds)}')
    writer.add_text(text_tag, f'Valid dataset size: {len(valid_ds)}')
    writer.add_text(text_tag, f'Number of classes: {len(train_ds.classes)}')

    train_loader = DataLoader(dataset=train_ds, batch_size=CONFIG.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=CONFIG.batch_size * 2)

    print('Start training')
    for epoch in range(CONFIG.epochs):
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                writer.add_scalars('loss', {'train': running_loss / 10}, epoch * len(train_loader) + i)
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                      (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                running_loss = 0.0

        pointnet.eval()
        correct = total = 0

        # validation
        if valid_loader:
            with torch.no_grad():
                for data in valid_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1, 2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            writer.add_scalars('accuracy', {'valid': val_acc}, epoch)
            print('Valid accuracy: %d %%' % val_acc)
        # save the model

        checkpoint = save_path / 'save_' + str(epoch) + '.pth'
        torch.save(pointnet.state_dict(), checkpoint)
        print('Model saved to ', checkpoint)


def probe_data():
    with open(CONFIG.dataset_path / "bed/train/bed_0001.off", 'r') as f:
        verts, faces = utils.read_off(f)
        i, j, k = np.array(faces).transpose()
        x, y, z = np.array(verts).transpose()
        verts_tensor = torch.from_numpy(np.vstack((x, y, z)).transpose()).unsqueeze(0)
        faces_tensor = torch.from_numpy(np.vstack((i, j, k)).transpose()).unsqueeze(0)
        writer.add_mesh('bed_mesh', vertices=verts_tensor, faces=faces_tensor)
        config_dict = {
            'material': {
                'cls': 'PointsMaterial',
                'size': 1
            }
        }
        writer.add_mesh('bed_point', vertices=verts_tensor, config_dict=config_dict)
        point_cloud = utils.PointSampler(3000)((verts, faces))
        point_cloud_tensor = torch.from_numpy(point_cloud).unsqueeze(0)
        writer.add_mesh('point_cloud', vertices=point_cloud_tensor, config_dict=config_dict)


def main():
    # open the tensorboard
    tensorboard_cmd = f'tensorboard --logdir {log_path} --port 0'
    pyperclip.copy(tensorboard_cmd)
    print(tensorboard_cmd)
    # probe_data()
    train()


if __name__ == '__main__':
    main()
