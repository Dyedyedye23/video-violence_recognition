import numpy as np
import torch
import cv2
from model import C3D_model
from torchvision.models.video import r3d_18
from torch import nn

def center_crop(frame):
    frame = frame[8:120, 30:142, :] # 裁剪帧的中心位置，大小112*112
    return np.array(frame).astype(np.uint8) # 将帧转换为数据类型为uint8的数组

def inference():
    # 定义模型训练的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载数据集标签
    with open("./data/labels.txt", 'r') as f:
        class_names = f.readlines()
        f.close()

    # # 加载模型，并将模型参数加载到模型中
    # model = C3D_model.C3D(num_classes=2)
    # checkpoint = torch.load('model_result2/models/best_model.pth.tar')
    # model.load_state_dict(checkpoint['state_dict'])

    # 加载预定义的 r3d_18 模型，并调整最后一层以适应你自己的分类任务
    model = r3d_18(pretrained=False)  # 不使用预训练权重
    model.fc = nn.Linear(model.fc.in_features, 2)  # 假设有 2 个类别，调整输出层

    # 加载你自己训练好的模型参数
    checkpoint = torch.load('model_result/models/best_model_epoch_12.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # 将模型放入到设备中，并设置验证模式
    model.to(device)
    model.eval()

    video = "WARNING_ GRAPHIC CONTENT - Video shows police in Buffalo, New York, shoving man to ground.mp4"
    cap = cv2.VideoCapture(video)
    # cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 目标窗口大小
    target_width = 800
    target_height = 600

    # 计算保持比例的窗口大小
    aspect_ratio = width / height
    if target_width / target_height > aspect_ratio:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    # 创建一个命名窗口并设置窗口大小
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('result', new_width, new_height)

    retaining = True

    clip = []
    while retaining:
        retaining, frame = cap.read() # 读取视频帧
        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]]) # 归一化处理
        clip.append(tmp) # 将处理后的帧加入到clip中

        if len(clip) == 16: # 当视频帧数量达到16时
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = inputs.to(device)

            with torch.no_grad():
                outputs = model.forward(inputs)
            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)

            clip.pop(0) # 移除列表中最早的帧，以便添加下一帧


        cv2.imshow('result', frame) # 显示推理结果
        cv2.waitKey(30)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    inference()