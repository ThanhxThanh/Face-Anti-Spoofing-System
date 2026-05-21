"""
realtime.py — Realtime Face Anti-Spoofing với webcam

Cải tiến so với version cũ:
    - Fix bug label mapping (false=0=spoof, true=1=live)
    - Thêm confidence threshold (chỉ hiện kết quả khi model đủ tự tin)
    - Thêm margin khi crop face (giống lúc preprocess)
    - Hiển thị confidence % trên màn hình
    - Smooth kết quả qua nhiều frame (tránh nhấp nháy)
    - Dùng PyTorch trực tiếp thay vì ONNX (dùng luôn model đã train)

Cài đặt:
    pip install opencv-python torch torchvision ultralytics

Chạy:
    python realtime.py
    Nhấn Q để thoát
"""

import cv2
import torch
import numpy as np
from collections import deque
from torchvision import transforms
from ultralytics import YOLO

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.mobilenetv3 import build_model

# ============================================================
# CONFIG
# ============================================================

# Đường dẫn model — dùng fine-tunbád model
#MODEL_PATH = "best_model_base.pth"
MODEL_PATH = "best_model_mixed.pth"
# Confidence threshold:
# Chỉ hiện kết quả khi model tự tin hơn ngưỡng này
# Nếu confidence thấp hơn → hiện "UNCERTAIN"
CONFIDENCE_THRESHOLD = 0.55  # 55% là ngưỡng hợp lý để giảm nhấp nháy nhưng vẫn k miss nhiều spoof

# Margin khi crop face (giống lúc preprocess)
FACE_MARGIN = 0.2

# Smooth qua N frame gần nhất để tránh nhấp nháy
SMOOTH_FRAMES = 10

# Kích thước input model
INPUT_SIZE = 224

# ============================================================
# SETUP
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Đang dùng: {DEVICE}")


def load_antispoof_model(path):
    """
    Load FAS model từ checkpoint PyTorch.
    Dùng trực tiếp thay vì export ONNX.
    """
    model = build_model().to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"✅ Loaded FAS model: {path}")
    print(f"   Class mapping: {checkpoint.get('class_map', 'N/A')}")
    # Class mapping: {'false': 0, 'true': 1}
    # → index 0 = spoof, index 1 = live
    return model


# ============================================================
# PREPROCESSING
# ============================================================

# Transform giống val_transform (không augment)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def crop_face_with_margin(frame, box, margin=FACE_MARGIN):
    """
    Crop mặt từ frame với margin.

    Args:
        frame : ảnh BGR từ webcam
        box   : [x1, y1, x2, y2] bounding box từ YOLO
        margin: tỷ lệ margin thêm vào (giống preprocess)

    Returns:
        face_rgb: numpy array RGB đã crop, hoặc None nếu crop lỗi
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    # Tính margin
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * margin)
    my = int(bh * margin)

    # Mở rộng bbox, clamp trong giới hạn frame
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    face_bgr = frame[y1:y2, x1:x2]

    if face_bgr.size == 0:
        return None, (x1, y1, x2, y2)

    # Convert BGR → RGB cho model
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return face_rgb, (x1, y1, x2, y2)


def predict(model, face_rgb):
    """
    Chạy inference trên 1 ảnh mặt.

    Returns:
        label     : "LIVE", "SPOOF", hoặc "UNCERTAIN"
        confidence: float 0.0 - 1.0
        live_prob : xác suất là live
        spoof_prob: xác suất là spoof
    """
    # Preprocess
    tensor = preprocess(face_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    spoof_prob = probs[0].item()
    live_prob  = probs[1].item()

    # Dead zone: nếu 2 class quá gần nhau → UNCERTAIN
    # Tránh nhấp nháy khi model không chắc
    diff = abs(live_prob - spoof_prob)
    if diff < 0.3:               # < 30% chênh lệch → không chắc
        return "UNCERTAIN", max(live_prob, spoof_prob), live_prob, spoof_prob

    if live_prob > spoof_prob:
        return "LIVE",  live_prob,  live_prob, spoof_prob
    else:
        return "SPOOF", spoof_prob, live_prob, spoof_prob


# ============================================================
# SMOOTHING
# Dùng deque để lưu N kết quả gần nhất
# Lấy kết quả xuất hiện nhiều nhất (majority voting)
# ============================================================

class ResultSmoother:
    """
    Smooth kết quả qua nhiều frame để tránh nhấp nháy.
    Dùng majority voting trên SMOOTH_FRAMES frame gần nhất.
    """
    def __init__(self, window_size=SMOOTH_FRAMES):
        self.window      = deque(maxlen=window_size)
        self.conf_window = deque(maxlen=window_size)

    def update(self, label, confidence):
        self.window.append(label)
        self.conf_window.append(confidence)

    def get_result(self):
        if not self.window:
            return "UNCERTAIN", 0.0

        # Majority voting
        from collections import Counter
        counts        = Counter(self.window)
        smooth_label  = counts.most_common(1)[0][0]
        smooth_conf   = np.mean(list(self.conf_window))
        return smooth_label, smooth_conf


# ============================================================
# DRAWING
# ============================================================

# Màu sắc cho từng kết quả
COLORS = {
    "LIVE"     : (0, 220, 0),    # xanh lá
    "SPOOF"    : (0, 0, 220),    # đỏ
    "UNCERTAIN": (0, 180, 220),  # vàng cam
}

def draw_result(frame, box, label, confidence, live_prob, spoof_prob):
    """
    Vẽ bounding box và kết quả lên frame.

    Hiển thị:
        - Bounding box với màu tương ứng
        - Label (LIVE / SPOOF / UNCERTAIN)
        - Confidence %
        - Thanh progress bar live/spoof probability
    """
    x1, y1, x2, y2 = box
    color = COLORS.get(label, (200, 200, 200))

    # Vẽ bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Background cho text
    text_main = f"{label}  {confidence*100:.1f}%"
    (tw, th), _ = cv2.getTextSize(text_main, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)

    # Label + confidence
    cv2.putText(frame, text_main,
                (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (255, 255, 255), 2)

    # Thanh probability bar bên dưới bbox
    bar_y    = y2 + 10
    bar_w    = x2 - x1
    bar_h    = 8

    # Background bar
    cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h),
                  (60, 60, 60), -1)

    # Live probability (xanh)
    live_w = int(bar_w * live_prob)
    cv2.rectangle(frame, (x1, bar_y), (x1 + live_w, bar_y + bar_h),
                  (0, 220, 0), -1)

    # Text live/spoof %
    cv2.putText(frame,
                f"L:{live_prob*100:.0f}%  S:{spoof_prob*100:.0f}%",
                (x1, bar_y + bar_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1)

    return frame


def draw_overlay(frame, fps):
    """Vẽ thông tin overlay góc trên bên trái."""
    h, w = frame.shape[:2]

    # Background mờ
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "Face Anti-Spoofing",
                (8, 22), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (255, 255, 255), 1)

    # FPS to và màu vàng dễ thấy
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (8, 52), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                (0, 255, 255), 2)  # màu vàng, chữ to hơn

    cv2.putText(frame, "Q: Quit",
                (200, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1)

    return frame


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print("\n🚀 Khởi động Face Anti-Spoofing realtime...")
    print(f"   Model              : {MODEL_PATH}")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"   Smooth frames      : {SMOOTH_FRAMES}")
    print(f"   Nhấn Q để thoát\n")

    # Load models
    fas_model = load_antispoof_model(MODEL_PATH)
    detector=  YOLO(r"G:\FPTU CAC KY\FPTU ki 5\DPL302m\ToanBoFinalProject\model.pt")# face detector
    print("✅ Loaded YOLO detector")

    # Khởi tạo smoother cho tối đa 10 khuôn mặt
    smoothers = {i: ResultSmoother() for i in range(10)}

    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không mở được webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # FPS counter
    import time
    prev_time = time.time()
    fps       = 0.0

    print("✅ Webcam sẵn sàng! Nhấn Q để thoát.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không đọc được frame từ webcam!")
            break

        # ── Detect faces ──
        results = detector(frame, imgsz=320, device=DEVICE,
                           verbose=False, conf=0.5)

        face_count = 0

        for r in results:
            for i, box in enumerate(r.boxes):
                # Crop face với margin
                face_rgb, crop_box = crop_face_with_margin(
                    frame, box.xyxy[0].cpu().numpy()
                )

                if face_rgb is None:
                    continue

                # Predict
                label, confidence, live_prob, spoof_prob = predict(
                    fas_model, face_rgb
                )

                # Smooth kết quả
                smoother_id = min(i, 9)
                smoothers[smoother_id].update(label, confidence)
                smooth_label, smooth_conf = smoothers[smoother_id].get_result()

                # Vẽ kết quả
                frame = draw_result(
                    frame, crop_box,
                    smooth_label, smooth_conf,
                    live_prob, spoof_prob
                )

                face_count += 1

        # ── FPS ──
        curr_time = time.time()
        fps       = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
        prev_time = curr_time

        # ── Overlay ──
        frame = draw_overlay(frame, fps)

        # ── Hiển thị ──
        cv2.imshow("Face Anti-Spoofing — FAS Demo", frame)

        # Thoát khi nhấn Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n👋 Thoát!")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()