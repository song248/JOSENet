import torch
import torch.nn.functional as F
import cv2
import os
import numpy as np
import yaml
from types import SimpleNamespace

from architectures import FGN_RGB, FGN_FLOW, FGN_MERGE_CLASSIFY, FGN
from preprocessing import generate_flow

# === 설정 ===
VIDEO_PATH = "test_02-1.mp4"  # 분석할 영상 파일 경로
MODEL_PATH = "models/primary/model_primary.pt"
CONFIG_PATH = "config/primary.yaml"
CLIP_FRAMES = 16
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# === config 불러오기 ===
def load_args(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    args = SimpleNamespace(**config)
    args.clip_frames = CLIP_FRAMES
    args.device = DEVICE
    return args


# === 전처리 함수 ===
def extract_clip(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total_frames // num_frames, 1)
    frames = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)
    cap.release()

    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))

    rgb_segment = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

    # Optical Flow 계산
    flow_segment = []
    for i in range(num_frames - 1):
        flow = generate_flow(frames[i], frames[i+1])  # shape: (H, W, 2)
        flow_segment.append(flow)
    flow_segment.append(np.zeros((IMG_SIZE, IMG_SIZE, 2)))  # 마지막 프레임 padding

    # 정규화
    rgb_segment = normalize(rgb_segment)
    flow_segment = normalize(flow_segment)

    rgb_tensor = torch.FloatTensor(np.transpose(np.array(rgb_segment), (3, 0, 1, 2)))  # [C, T, H, W]
    flow_tensor = torch.FloatTensor(np.transpose(np.array(flow_segment), (3, 0, 1, 2)))

    return rgb_tensor.unsqueeze(0), flow_tensor.unsqueeze(0)  # [1, C, T, H, W]


def normalize(data):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std if std > 0 else data - mean


# === 모델 로드 ===
def load_model(model_path, args):
    model_rgb = FGN_RGB(args)
    model_flow = FGN_FLOW(args)
    model_merge = FGN_MERGE_CLASSIFY(args)
    model = FGN(model_rgb, model_flow, model_merge)
    model = torch.nn.DataParallel(model).to(args.device)

    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()
    return model


# === 추론 ===
def predict(video_path):
    args = load_args(CONFIG_PATH)
    rgb_input, flow_input = extract_clip(video_path, num_frames=args.clip_frames)
    rgb_input = rgb_input.to(args.device)
    flow_input = flow_input.to(args.device)

    model = load_model(MODEL_PATH, args)

    with torch.no_grad():
        logits = torch.squeeze(model(rgb_input, flow_input))
        prob = torch.sigmoid(logits).item()

    print(f"\n🎞️ 영상: {os.path.basename(video_path)}")
    print(f"🔥 Fight 확률: {prob * 100:.2f}%")
    print(f"🧾 예측 결과: {'Fight 🔥' if prob >= 0.5 else 'NonFight ✅'}")


# === 실행 ===
if __name__ == "__main__":
    predict(VIDEO_PATH)
