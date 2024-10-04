import numpy as np
import torch
import cv2
import torch.nn as nn
from torchvision.models.video import r3d_18  # 导入 PyTorch 自带的 r3d_18 模型


# 中心裁剪函数
def center_crop(frame):
    return frame[8:120, 30:142, :]  # 裁剪帧的中心位置，大小112*112


# 推理函数
def inference():
    # 定义设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载标签
    with open("./data/labels.txt", 'r') as f:
        class_names = [line.strip() for line in f]

    # 加载预定义的 r3d_18 模型，并调整最后一层以适应你自己的分类任务
    model = r3d_18(pretrained=False)  # 不使用预训练权重
    model.fc = nn.Linear(model.fc.in_features, 2)  # 假设有 2 个类别，调整输出层

    # 加载你自己训练好的模型参数
    checkpoint = torch.load('model_result/models/best_model_epoch_12.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # 打开视频文件
    video = "Gandhi (1982) Trailer #1 _ Movieclips Classic Trailers.mp4"
    cap = cv2.VideoCapture(video)

    # 设置目标窗口大小
    target_width, target_height = 800, 600
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = width / height
    if target_width / target_height > aspect_ratio:
        new_width, new_height = int(target_height * aspect_ratio), target_height
    else:
        new_width, new_height = target_width, int(target_width / aspect_ratio)

    # 创建显示窗口
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('result', new_width, new_height)

    clip = []
    retaining = True

    while retaining:
        retaining, frame = cap.read()  # 读取视频帧
        if not retaining and frame is None:
            continue

        # 预处理帧
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))  # 先调整大小再裁剪
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])  # 归一化
        clip.append(tmp)

        if len(clip) == 16:  # 收集16帧
            inputs = np.array(clip).astype(np.float32)
            inputs = np.transpose(inputs, (0, 3, 1, 2))  # 调整为 (帧数, 通道数, 高度, 宽度)
            inputs = torch.from_numpy(inputs).unsqueeze(0).to(device)  # 添加批次维度并转为 tensor

            with torch.no_grad():
                outputs = model(inputs)  # 使用 r3d_18 模型进行推理
            probs = torch.nn.Softmax(dim=1)(outputs)  # 计算类别概率
            label = torch.argmax(probs, dim=1).item()

            # 在帧上显示类别与概率
            cv2.putText(frame, class_names[label], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.putText(frame, f"prob: {probs[0][label]:.4f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            clip.pop(0)  # 移除最早的帧

        # 显示推理结果
        cv2.imshow('result', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    inference()
