import torch
import cv2
import numpy as np
import argparse
import os
from face_detector.model import SSD
from face_detector.dataset import default_box

IMAGE_LABELS = ['background', 'face']


def decode_bbox_torch(predicts, boxes, variances=[0.1, 0.2]):
    """예측된 바운딩 박스 디코딩"""
    centers = boxes[:, :2] + predicts[:, :2] * variances[0] * boxes[:, 2:]
    sides = boxes[:, 2:] * torch.exp(predicts[:, 2:] * variances[1])
    return torch.cat([centers - sides / 2, centers + sides / 2], dim=1)


def parse_predict(predictions, boxes, score_threshold=0.5, nms_threshold=0.05):
    """모델 예측을 파싱하여 박스, 클래스, 점수 획득"""
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

        # 고쳐보다가 파이토치 nms 사용
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
    """이미지를 max_steps로 나누어 떨어지도록 패딩"""
    img_h, img_w, _ = img.shape

    img_pad_h = (max_steps - img_h % max_steps) % max_steps
    img_pad_w = (max_steps - img_w % max_steps) % max_steps

    padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)

    img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                             cv2.BORDER_CONSTANT, value=padd_val.tolist())

    pad_params = (img_h, img_w, img_pad_h, img_pad_w)

    return torch.from_numpy(img).permute(2, 0, 1).float(), pad_params


def recover_pad(boxes, pad_params):
    """패딩 후 박스 좌표 복구"""
    img_h, img_w, img_pad_h, img_pad_w = pad_params
    if img_pad_h == 0 and img_pad_w == 0:
        return boxes

    # 박스 줄였다가 리사이즈하면서 원래 사이즈로 복원
    scale_x = img_w / (img_w + img_pad_w)
    scale_y = img_h / (img_h + img_pad_h)
    box = np.reshape(boxes, [-1, 2, 2]) * [scale_x, scale_y]
    boxes = np.reshape(box, [-1, 4])
    return boxes


def draw_boxes(img, boxes, classes, scores, class_list, dunce_img=None):
    """이미지에 바운딩 박스와 고깔 모자(dunce hat) 오버레이 그리기"""
    img_height, img_width = img.shape[:2]

    for i in range(len(boxes)):
        # 절대값
        if boxes[i].max() <= 1.0:
            # 정규화 좌표
            x_min = int(boxes[i][0] * img_width)
            y_min = int(boxes[i][1] * img_height)
            x_max = int(boxes[i][2] * img_width)
            y_max = int(boxes[i][3] * img_height)
        else:
            # 절대 좌표
            x_min = int(boxes[i][0])
            y_min = int(boxes[i][1])
            x_max = int(boxes[i][2])
            y_max = int(boxes[i][3])

        if classes[i] == 1:
            color = (0, 255, 0)  # 초록
        else:
            color = (0, 0, 255)  # 빨강

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        if len(scores) > i:
            score = "{:.4f}".format(scores[i])
            class_name = class_list[classes[i]]
            label = '{} {}'.format(class_name, score)
            position = (x_min, y_min - 4)
            cv2.putText(img, label, position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 얼굴 바운딩 박스 위에 고깔 모자 그리기
        if dunce_img is not None and classes[i] == 1:  # 얼굴인 경우에만
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            # bbox 위에 맞게 고깔 이미지 리사이즈 (너비는 bbox 너비와 일치)
            hat_width = int(bbox_width * 1.2)  # 얼굴보다 약간 넓게
            hat_aspect_ratio = dunce_img.shape[1] / dunce_img.shape[0]
            hat_height = int(hat_width / hat_aspect_ratio)

            dunce_resized = cv2.resize(dunce_img, (hat_width, hat_height))

            # 위치: 바운딩 박스 위 중앙
            hat_x = x_min - (hat_width - bbox_width) // 2
            hat_y = y_min - hat_height

            # 모자가 이미지 경계 내에 있는지 확인
            if hat_y >= 0 and hat_x >= 0 and hat_x + hat_width <= img_width:
                # PNG에 알파 채널이 있는 경우 알파 블렌딩으로 오버레이
                if dunce_resized.shape[2] == 4:  # RGBA
                    alpha = dunce_resized[:, :, 3] / 255.0
                    for c in range(3):  # BGR 채널
                        img[hat_y:hat_y+hat_height, hat_x:hat_x+hat_width, c] = \
                            alpha * dunce_resized[:, :, c] + \
                            (1 - alpha) * img[hat_y:hat_y +
                                              hat_height, hat_x:hat_x+hat_width, c]
                else:  # 알파 채널 없음, 그냥 덮어쓰기
                    img[hat_y:hat_y+hat_height, hat_x:hat_x +
                        hat_width] = dunce_resized[:, :, :3]

    return img


def inference(model, image_path, boxes, device, output_path=None, score_threshold=0.9, image_height=256, image_width=320):
    """단일 이미지에 대해 추론 실행"""
    # 고깔 모자 이미지 로드
    dunce_path = os.path.join(os.path.dirname(
        __file__), 'face_detector', 'goat.png')
    dunce_img = None
    if os.path.exists(dunce_path):
        # 알파 채널 보존을 위해 IMREAD_UNCHANGED 사용
        dunce_img = cv2.imread(dunce_path, cv2.IMREAD_UNCHANGED)
        if dunce_img is not None:
            print(f"Loaded dunce hat from {dunce_path}")
    else:
        print(f"Warning: dunce.png not found at {dunce_path}")

    # 이미지 읽기 및 전처리
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # 최종 시각화를 위해 원본 크기 저장
    original_img = img_raw.copy()

    # 모델 입력 크기로 리사이즈 (320x256)
    img = cv2.resize(img_raw, (image_width, image_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img.copy())

    # 이미지 패딩
    img_tensor, pad_params = pad_input_image(img, max_steps=64)
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)  # 배치 차원 추가

    # 추론 실행
    model.eval()
    with torch.no_grad():
        predictions = model(img_tensor)

    # 예측 파싱
    pred_boxes, labels, scores = parse_predict(
        predictions, boxes, score_threshold=score_threshold)

    if len(pred_boxes) == 0:
        print("No faces detected.")
        return original_img

    # 패딩 복구
    pred_boxes = recover_pad(pred_boxes, pad_params)

    # 박스가 경계를 벗어나지 않도록 [0, 1] 범위로 클리핑
    pred_boxes = np.clip(pred_boxes, 0.0, 1.0)

    # 박스를 원본 이미지 크기로 스케일링 (절대 좌표)
    orig_h, orig_w = original_img.shape[:2]
    pred_boxes[:, [0, 2]] *= orig_w  # x 좌표
    pred_boxes[:, [1, 3]] *= orig_h  # y 좌표

    # 원본 이미지에 박스와 고깔 모자 그리기 (절대 좌표)
    result_img = draw_boxes(original_img.copy(), pred_boxes,
                            labels, scores, IMAGE_LABELS, dunce_img)

    # 저장 또는 표시
    if output_path:
        cv2.imwrite(output_path, result_img)
        print(f"Result saved to {output_path}")

    print(f"Detected {len(pred_boxes)} face(s)")

    return result_img


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 모델 로드
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 320

    model = SSD(num_classes=len(IMAGE_LABELS),
                input_shape=(3, IMAGE_HEIGHT, IMAGE_WIDTH))

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {args.checkpoint}")

    # 기본 박스(anchor box) 로드
    boxes = default_box(IMAGE_HEIGHT, IMAGE_WIDTH).to(device)
    boxes = boxes.to(torch.float32)

    # 추론 실행
    if os.path.isdir(args.input):
        # 디렉토리 내 모든 이미지 처리
        output_dir = args.output if args.output else os.path.join(
            args.input, 'inference_results')
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(args.input, filename)
                output_path = os.path.join(output_dir, f'detected_{filename}')
                print(f"\nProcessing {filename}...")
                inference(model, image_path, boxes, device,
                          output_path, args.score_threshold)
    else:
        # 단일 이미지 처리
        if args.output:
            output_path = args.output
        else:
            # 파일 확장자 처리
            root, ext = os.path.splitext(args.input)
            output_path = f"{root}_detected{ext}"
        result = inference(model, args.input, boxes, device,
                           output_path, args.score_threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SSD 얼굴 감지 추론')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='모델 체크포인트 경로 (.pth 파일)')
    parser.add_argument('--input', type=str, required=True,
                        help='입력 이미지 또는 디렉토리 경로')
    parser.add_argument('--output', type=str, default=None,
                        help='출력 이미지 또는 디렉토리 경로 (선택 사항)')
    parser.add_argument('--score_threshold', type=float,
                        default=0.5, help='감지 임계값')

    args = parser.parse_args()
    main(args)
