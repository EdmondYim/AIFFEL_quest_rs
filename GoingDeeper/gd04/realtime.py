import cv2
import torch
import numpy as np
import argparse
import time
import os
from face_detector.model import SSD
from face_detector.dataset import default_box


'''
# Real-time Face Detection

이 스크립트는 웹캠을 사용하여 실시간으로 얼굴을 감지합니다.

## 사용 방법

### 기본 실행 (기본 체크포인트 사용)
```bash
python realtime.py
```

### 특정 체크포인트 지정
```bash
python realtime.py --checkpoint checkpoints/ssd_epoch_65.pth
```

### 감지 임계값 조정
```bash
python realtime.py --checkpoint checkpoints_hardaug/ssd_epoch_80.pth --score_threshold 0.97
python realtime.py --checkpoint checkpoints_small_aug/ssd_epoch_80.pth --score_threshold 0.97
python realtime.py --checkpoint checkpoints/ssd_epoch_80.pth --score_threshold 0.97
```

## 매개변수

- `--checkpoint`: 모델 체크포인트 파일 경로 (기본값: `checkpoints/best_checkpoint.pth`)
- `--score_threshold`: 감지 임계값 (기본값: 0.5)
  - 높은 값 (0.7~0.9): 더 확실한 얼굴만 감지 (false positive 감소)
  - 낮은 값 (0.3~0.5): 더 많은 얼굴 감지 (false positive 증가 가능)


'''

IMAGE_LABELS = ['background', 'face']


def decode_bbox_torch(predicts, boxes, variances=[0.1, 0.2]):
    centers = boxes[:, :2] + predicts[:, :2] * variances[0] * boxes[:, 2:]
    sides = boxes[:, 2:] * torch.exp(predicts[:, 2:] * variances[1])
    return torch.cat([centers - sides / 2, centers + sides / 2], dim=1)


def parse_predict(predictions, boxes, score_threshold=0.5, nms_threshold=0.03):
    bbox_predictions, confidences = torch.split(
        predictions[0], [4, predictions.size(-1) - 4], dim=-1)
    boxes_decoded = decode_bbox_torch(bbox_predictions, boxes)

    scores = torch.softmax(confidences, dim=-1)

    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, len(IMAGE_LABELS)):
        cls_scores = scores[:, c]

        score_idx = cls_scores > score_threshold

        cls_boxes = boxes_decoded[score_idx]
        cls_scores = cls_scores[score_idx]

        if cls_boxes.size(0) == 0:
            continue

        from torchvision.ops import nms
        keep = nms(cls_boxes, cls_scores, nms_threshold)

        cls_boxes = cls_boxes[keep]
        cls_scores = cls_scores[keep]

        cls_labels = [c] * cls_boxes.size(0)

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)

    if len(out_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    out_boxes = torch.cat(out_boxes, dim=0)
    out_scores = torch.cat(out_scores, dim=0)

    boxes = torch.clamp(out_boxes, 0.0, 1.0).cpu().numpy()
    classes = np.array(out_labels)
    scores = out_scores.cpu().numpy()

    return boxes, classes, scores


def pad_input_image(img, max_steps):
    img_h, img_w, _ = img.shape

    img_pad_h = (max_steps - img_h % max_steps) % max_steps
    img_pad_w = (max_steps - img_w % max_steps) % max_steps

    padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)

    img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                             cv2.BORDER_CONSTANT, value=padd_val.tolist())

    pad_params = (img_h, img_w, img_pad_h, img_pad_w)

    return torch.from_numpy(img).permute(2, 0, 1).float(), pad_params


def recover_pad(boxes, pad_params):
    """Recover boxes after padding"""
    img_h, img_w, img_pad_h, img_pad_w = pad_params
    if img_pad_h == 0 and img_pad_w == 0:
        return boxes

    scale_x = img_w / (img_w + img_pad_w)
    scale_y = img_h / (img_h + img_pad_h)
    box = np.reshape(boxes, [-1, 2, 2]) * [scale_x, scale_y]
    boxes = np.reshape(box, [-1, 4])
    return boxes


def draw_boxes(img, boxes, classes, scores, class_list, dunce_img=None):

    img_height, img_width = img.shape[:2]

    for i in range(len(boxes)):

        if boxes[i].max() <= 1.0:

            x_min = int(boxes[i][0] * img_width)
            y_min = int(boxes[i][1] * img_height)
            x_max = int(boxes[i][2] * img_width)
            y_max = int(boxes[i][3] * img_height)
        else:

            x_min = int(boxes[i][0])
            y_min = int(boxes[i][1])
            x_max = int(boxes[i][2])
            y_max = int(boxes[i][3])

        if classes[i] == 1:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        if len(scores) > i:
            score = "{:.4f}".format(scores[i])
            class_name = class_list[classes[i]]
            label = '{} {}'.format(class_name, score)
            position = (x_min, y_min - 4)
            cv2.putText(img, label, position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 스티커 올리기
        if dunce_img is not None and classes[i] == 1:  # 얼굴만
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            # 리사이징
            hat_width = int(bbox_width * 1.2)
            hat_aspect_ratio = dunce_img.shape[1] / dunce_img.shape[0]
            hat_height = int(hat_width / hat_aspect_ratio)

            dunce_resized = cv2.resize(dunce_img, (hat_width, hat_height))

            hat_x = x_min - (hat_width - bbox_width) // 2
            hat_y = y_min - hat_height

            if hat_y >= 0 and hat_x >= 0 and hat_x + hat_width <= img_width:

                if dunce_resized.shape[2] == 4:  # RGBA
                    alpha = dunce_resized[:, :, 3] / 255.0
                    for c in range(3):
                        img[hat_y:hat_y+hat_height, hat_x:hat_x+hat_width, c] = \
                            alpha * dunce_resized[:, :, c] + \
                            (1 - alpha) * img[hat_y:hat_y +
                                              hat_height, hat_x:hat_x+hat_width, c]
                else:
                    img[hat_y:hat_y+hat_height, hat_x:hat_x +
                        hat_width] = dunce_resized[:, :, :3]

    return img


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 스티커 불러오기
    dunce_path = os.path.join(os.path.dirname(
        __file__), 'face_detector', 'goat.png')  # dunce.png
    dunce_img = None
    if os.path.exists(dunce_path):
        dunce_img = cv2.imread(dunce_path, cv2.IMREAD_UNCHANGED)
        if dunce_img is not None:
            print(f"스티커 이름,경로 확인 {dunce_path}")

    # 모델 불러오기
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 320

    model = SSD(num_classes=len(IMAGE_LABELS),
                input_shape=(3, IMAGE_HEIGHT, IMAGE_WIDTH))

    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Loaded model from {args.checkpoint}")
    except Exception as yoyoyoyoyo:
        print(f"Error loading model: {yoyoyoyoyo}")
        return
    # 예상인데, default_box를 얼굴에 맞게 건드리거나 데이터셋을 변경하면 좀더 성능이 좋을 것 같습니다.
    boxes = default_box(IMAGE_HEIGHT, IMAGE_WIDTH).to(device)
    boxes = boxes.to(torch.float32)

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("q 눌러서 나가기")

    # FPS calculation
    fps_time = time.time()
    fps_counter = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # ret  : boolean / frame :  사진
        original_frame = frame.copy()

        # 리사이징  / 임보혁 깃헙 ex04 참조
        img = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img.copy())

        # Pad image
        img_tensor, pad_params = pad_input_image(img, max_steps=64)
        img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(img_tensor)

        pred_boxes, labels, scores = parse_predict(
            predictions, boxes, score_threshold=args.score_threshold)

        if len(pred_boxes) > 0:

            pred_boxes = recover_pad(pred_boxes, pad_params)

            pred_boxes = np.clip(pred_boxes, 0.0, 1.0)

            original_frame = draw_boxes(
                original_frame, pred_boxes, labels, scores, IMAGE_LABELS, dunce_img)

        # 프레임 카운터
        fps_counter += 1
        if time.time() - fps_time > 1:
            fps = fps_counter / (time.time() - fps_time)
            fps_counter = 0
            fps_time = time.time()

        # 녹색글자 보여주기
        cv2.putText(original_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Real-time Face Detection', original_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("끝.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SSD Real-time Face Detection')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/ssd_epoch_80.pth',
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--score_threshold', type=float, default=0.97,
                        help='Score threshold for detection')

    args = parser.parse_args()
    main(args)
