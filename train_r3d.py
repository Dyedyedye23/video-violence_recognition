import socket
import torch
from torch import nn, optim
from torchvision.models.video import r3d_18
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import timeit
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import VideoDataset


def train_model(num_epochs, num_classes, lr, device, save_dir, train_dataloader, val_dataloader, test_dataloader):
    # 使用预训练的R3D-18模型
    model = r3d_18(pretrained=True)

    # 修改最后一层以适应当前的num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 冻结所有层的参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻最后几层卷积层以及全连接层
    # 这里假设解冻最后两层卷积层和全连接层
    for name, param in model.layer4.named_parameters():
        param.requires_grad = True  # 解冻 layer4 卷积层
    for param in model.fc.parameters():
        param.requires_grad = True  # 解冻全连接层

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器：只优化解冻的层，使用 Adam 优化器
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,  # 建议较小的学习率，比如 1e-4
        weight_decay=5e-4
    )

    # 定义学习率的更新策略
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 将模型和损失函数放入设备
    model.to(device)
    criterion.to(device)

    # 日志记录
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # 训练过程
    data_loaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {x: len(data_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()
            running_loss = 0.0
            running_corrects = 0.0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(data_loaders[phase]):
                inputs = inputs.to(device).requires_grad_()
                labels = labels.long().to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            writer.add_scalar(f'data/{phase}_loss_epoch', epoch_loss, epoch)
            writer.add_scalar(f'data/{phase}_acc_epoch', epoch_acc, epoch)

            stop_time = timeit.default_timer()
            print(f"[{phase}] Epoch: {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            print(f"Execution time: {stop_time - start_time}\n")

            # 保存最佳验证模型
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                }, os.path.join(save_dir, 'models', f'best_model_epoch_{epoch + 1}.pth.tar'))
                print(f"Best model saved with validation accuracy: {best_val_acc:.4f}\n")

            # 每10轮保存一次模型
            if epoch % 10 == 9:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, os.path.join(save_dir, 'models', f'r3d_epoch_{epoch + 1}.pth.tar'))
                print(f"Model saved at epoch {epoch + 1}\n")

    writer.close()

    # 模型测试
    model.eval()
    running_corrects = 0.0
    for inputs, labels in tqdm(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.long().to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / test_size
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 30
    num_classes = 2
    lr = 1e-4
    save_dir = 'model_result'

    train_data = VideoDataset(dataset_path='data/violence_data', images_path='train', clip_len=16)
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)

    val_data = VideoDataset(dataset_path='data/violence_data', images_path='val', clip_len=16)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True, num_workers=2)

    test_data = VideoDataset(dataset_path='data/violence_data', images_path='test', clip_len=16)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=2)

    train_model(num_epochs, num_classes, lr, device, save_dir, train_dataloader, val_dataloader, test_dataloader)
