"""
Face Anti-Spoofing Realtime Demo
Requirements:
    pip install opencv-python torch torchvision ultralytics

Usage:
    python realtime.py
    Press Q to quit
"""

import cv2
import time
import torch
import numpy as np
from collections import deque, Counter
from torchvision import transforms
from ultralytics import YOLO

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.mobilenetv3 import build_model

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "best_model_mixed.pth"
FACE_DET_PATH = r"G:\FPTU CAC KY\FPTU ki 5\DPL302m\ToanBoFinalProject\model.pt"

# Filter 1: minimum gap between live/spoof probability to avoid uncertain zone
FAS_DEAD_ZONE = 0.30

# Filter 2: minimum confidence to show a prediction
CONFIDENCE_THRESHOLD = 0.55

# Margin ratio added around face bounding box (matches preprocessing)
FACE_MARGIN = 0.2

# Number of recent frames for majority voting smoothing
SMOOTH_FRAMES = 15

# Number of consecutive LIVE frames required before confirming LIVE
LIVE_CONFIRM_FRAMES = 10

# Model input resolution
INPUT_SIZE = 224

# ============================================================
# SETUP
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def load_antispoof_model(path):
    """Load FAS model from PyTorch checkpoint."""
    model = build_model().to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded FAS model: {path}")
    print(f"Class mapping: {checkpoint.get('class_map', 'N/A')}")
    # Class mapping: {'false': 0, 'true': 1} → index 0 = spoof, index 1 = live
    return model


# ============================================================
# PREPROCESSING
# ============================================================

# Validation transform — no augmentation
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def crop_face_with_margin(frame, box, margin=FACE_MARGIN):
    """
    Crop face region from frame with added margin.

    Args:
        frame  : BGR image from webcam
        box    : [x1, y1, x2, y2] bounding box from YOLO
        margin : margin ratio to expand bounding box

    Returns:
        face_rgb : RGB numpy array of cropped face, or None if invalid
        crop_box : adjusted bounding box (x1, y1, x2, y2)
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    face_bgr = frame[y1:y2, x1:x2]
    if face_bgr.size == 0:
        return None, (x1, y1, x2, y2)

    return cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB), (x1, y1, x2, y2)


def predict(model, face_rgb):
    """
    Run inference on a single face crop.

    Returns:
        label      : "LIVE", "SPOOF", or "UNCERTAIN"
        confidence : float between 0.0 and 1.0
        live_prob  : probability of being a live face
        spoof_prob : probability of being a spoof
    """
    tensor = preprocess(face_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    spoof_prob = probs[0].item()
    live_prob  = probs[1].item()

    # Filter 1: dead zone — too close to call
    if abs(live_prob - spoof_prob) < FAS_DEAD_ZONE:
        return "UNCERTAIN", max(live_prob, spoof_prob), live_prob, spoof_prob

    label      = "LIVE" if live_prob > spoof_prob else "SPOOF"
    confidence = live_prob if label == "LIVE" else spoof_prob

    # Filter 2: confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        label = "UNCERTAIN"

    return label, confidence, live_prob, spoof_prob


# ============================================================
# FACE TRACKER
# Tracks per-face state including smoothing and LIVE confirmation
# ============================================================

class FaceTracker:
    """
    Tracks prediction state for a single face across frames.

    Combines majority-vote smoothing with a consecutive LIVE streak
    counter to avoid triggering on brief flickers.
    """

    def __init__(self, window=SMOOTH_FRAMES):
        self.labels      = deque(maxlen=window)
        self.confs       = deque(maxlen=window)
        self.live_streak = 0
        self.confirmed   = False

    def update(self, label, conf):
        self.labels.append(label)
        self.confs.append(conf)

        if label == "LIVE":
            self.live_streak += 1
        else:
            # Reset confirmation on any non-LIVE frame
            self.live_streak = 0
            self.confirmed   = False

        if self.live_streak >= LIVE_CONFIRM_FRAMES:
            self.confirmed = True

    def get_result(self):
        """Return smoothed label and average confidence."""
        if not self.labels:
            return "UNCERTAIN", 0.0
        label = Counter(self.labels).most_common(1)[0][0]
        conf  = float(np.mean(self.confs))
        return label, conf

    @property
    def confirm_progress(self):
        """Confirmation progress as a ratio from 0.0 to 1.0."""
        return min(self.live_streak / LIVE_CONFIRM_FRAMES, 1.0)


# ============================================================
# DRAWING
# ============================================================

COLORS = {
    "LIVE"     : (0, 220, 0),
    "SPOOF"    : (0, 0, 220),
    "UNCERTAIN": (0, 180, 220),
}


def draw_result(frame, box, label, confidence, live_prob, spoof_prob, tracker):
    """
    Draw bounding box, prediction label, probability bar,
    and LIVE confirmation progress onto frame.
    """
    x1, y1, x2, y2 = box
    color = COLORS.get(label, (200, 200, 200))

    # Bounding box — thicker when confirmed LIVE
    thickness = 3 if tracker.confirmed else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label + confidence
    text = f"{label}  {confidence*100:.1f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, text, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

    # Live/Spoof probability bar
    bar_y, bar_w, bar_h = y2 + 10, x2 - x1, 8
    cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x1, bar_y),
                  (x1 + int(bar_w * live_prob), bar_y + bar_h), (0, 220, 0), -1)
    cv2.putText(frame, f"L:{live_prob*100:.0f}%  S:{spoof_prob*100:.0f}%",
                (x1, bar_y + bar_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # LIVE confirmation progress bar
    if label == "LIVE" and not tracker.confirmed:
        prog_y = y2 + 32
        prog_w = int(bar_w * tracker.confirm_progress)
        cv2.rectangle(frame, (x1, prog_y), (x2, prog_y + 5), (40, 40, 40), -1)
        cv2.rectangle(frame, (x1, prog_y), (x1 + prog_w, prog_y + 5),
                      (0, 255, 200), -1)
        frames_left = LIVE_CONFIRM_FRAMES - tracker.live_streak
        cv2.putText(frame, f"Verifying... ({frames_left} frames left)",
                    (x1, prog_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1)

    # Confirmed LIVE badge
    if tracker.confirmed:
        cv2.putText(frame, "CONFIRMED LIVE",
                    (x1, y2 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)

    return frame


def draw_overlay(frame, fps):
    """Draw FPS and info overlay on the top-left corner."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "Face Anti-Spoofing",
                (8, 22), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (8, 52), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "Q: Quit",
                (200, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return frame


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print("Starting Face Anti-Spoofing realtime...")
    print(f"  Model               : {MODEL_PATH}")
    print(f"  Dead zone           : {FAS_DEAD_ZONE}")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"  Smooth frames       : {SMOOTH_FRAMES}")
    print(f"  LIVE confirm frames : {LIVE_CONFIRM_FRAMES}")
    print(f"  Press Q to quit\n")

    fas_model = load_antispoof_model(MODEL_PATH)
    detector  = YOLO(FACE_DET_PATH)
    print("Loaded YOLO detector")

    # One tracker per face slot (max 5 faces)
    trackers = {i: FaceTracker() for i in range(5)}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # Lower resolution for higher FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()
    fps = 0.0

    print("Webcam ready. Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from webcam.")
            break

        # Lower imgsz for faster YOLO inference
        results = detector(frame, imgsz=256, device=DEVICE, verbose=False, conf=0.5)

        for r in results:
            for i, box in enumerate(r.boxes):
                tracker = trackers[min(i, 4)]

                face_rgb, crop_box = crop_face_with_margin(
                    frame, box.xyxy[0].cpu().numpy()
                )
                if face_rgb is None:
                    continue

                label, confidence, live_prob, spoof_prob = predict(fas_model, face_rgb)

                tracker.update(label, confidence)
                smooth_label, smooth_conf = tracker.get_result()

                frame = draw_result(
                    frame, crop_box,
                    smooth_label, smooth_conf,
                    live_prob, spoof_prob,
                    tracker
                )

        curr_time = time.time()
        fps       = 0.9 * fps + 0.1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        frame = draw_overlay(frame, fps)
        cv2.imshow("Face Anti-Spoofing — FAS Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()