

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os

from model import RetinaNet, DecodePredictions, get_resnet50_backbone


# 클래스별 색상
CLASS_COLORS = {
    'Car': (0, 0, 255),           # 빨강 
    'Van': (0, 128, 255),         # 주황
    'Truck': (0, 255, 255),       # 노랑
    'Pedestrian': (0, 255, 0),    # 초록
    'Person_sitting': (128, 255, 0),
    'Cyclist': (255, 128, 0),
    'Tram': (255, 0, 128),
    'Misc': (255, 0, 255)
}


def load_model(checkpoint_path='checkpoints/best.pth', num_classes=8, device='cuda'):
    """모델 로드"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    backbone = get_resnet50_backbone(pretrained=False)
    model = RetinaNet(num_classes=num_classes, backbone=backbone)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"모델 로드: {checkpoint_path}")
    except FileNotFoundError:
        print(f"체크포인트 없음: {checkpoint_path}")
        print("랜덤 초기화 모델 사용")
    
    model = model.to(device)
    model.eval()
    
    decoder = DecodePredictions(
        num_classes=num_classes,
        confidence_threshold=0.4,
        nms_iou_threshold=0.5,
        max_detections=100
    )
    
    return model, decoder, device


def process_frame(frame, model, decoder, device, img_size=(384, 1280)):
    """
    단일 프레임 처리
    
    Args:
        frame: OpenCV 프레임 (BGR, numpy array)
        model: RetinaNet 모델
        decoder: DecodePredictions
        device: torch device
        img_size: 모델 입력 크기
    
    Returns:
        boxes, scores, classes, (scale_w, scale_h)
    """
    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # 원본 크기
    original_size = pil_image.size  # (width, height)
    scale_w = img_size[1] / original_size[0]
    scale_h = img_size[0] / original_size[1]
    
    # 전처리
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # 추론
    with torch.no_grad():
        predictions = model(image_tensor)
        decoded = decoder(image_tensor, predictions)
    
    boxes, scores, classes = decoded[0]
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    classes = classes.cpu().numpy()
    
    return boxes, scores, classes, (scale_w, scale_h)


def draw_detections(frame, boxes, scores, classes, scales, 
                    class_names, show_scores=True, min_score=0.3):
    """
    프레임에 탐지 결과 그리기
    
    Args:
        frame: OpenCV 프레임 (BGR)
        boxes: 바운딩 박스 배열
        scores: 점수 배열
        classes: 클래스 배열
        scales: (scale_w, scale_h)
        class_names: 클래스 이름 리스트
        show_scores: 점수 표시 여부
        min_score: 최소 표시 점수
    
    Returns:
        그려진 프레임, 통계 정보
    """
    scale_w, scale_h = scales
    output = frame.copy()
    
    class_counts = {}
    detection_count = 0
    
    for box, score, cls in zip(boxes, scores, classes):
        if score < min_score:
            continue
        
        detection_count += 1
        
        # 원본 좌표로 변환
        x1 = int(box[0] / scale_w)
        y1 = int(box[1] / scale_h)
        x2 = int(box[2] / scale_w)
        y2 = int(box[3] / scale_h)
        
        class_name = class_names[cls]
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # 바운딩 박스
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # 라벨
        if show_scores:
            label = f"{class_name} {score:.2f}"
        else:
            label = class_name
        
        # 라벨 배경
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(output, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), color, -1)
        cv2.putText(output, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    stats = {
        'total': detection_count,
        'classes': class_counts
    }
    
    return output, stats


def check_stop_condition(boxes, scores, classes, scales, class_names, size_limit=300):
    """
    정지 조건 확인 (self_drive_assist 로직)
    
    Returns:
        should_stop (bool), reason (str)
    """
    scale_w, scale_h = scales
    
    person_classes = {'Pedestrian', 'Person_sitting', 'Cyclist'}
    vehicle_classes = {'Car', 'Van', 'Truck'}
    
    person_indices = [i for i, name in enumerate(class_names) if name in person_classes]
    vehicle_indices = [i for i, name in enumerate(class_names) if name in vehicle_classes]
    
    # 조건 1: 사람 탐지
    for cls in classes:
        if cls in person_indices:
            return True, f"Person detected ({class_names[cls]})"
    
    # 조건 2: 큰 차량
    for box, cls in zip(boxes, classes):
        if cls in vehicle_indices:
            x1, y1, x2, y2 = box
            x1_orig = x1 / scale_w
            y1_orig = y1 / scale_h
            x2_orig = x2 / scale_w
            y2_orig = y2 / scale_h
            
            width = x2_orig - x1_orig
            height = y2_orig - y1_orig
            
            if width >= size_limit or height >= size_limit:
                return True, f"Large vehicle ({class_names[cls]}: {max(width, height):.0f}px)"
    
    return False, "Safe"


def process_video(
    video_path,
    output_path='output1.mp4',
    checkpoint_path='checkpoints/best.pth',
    img_size=(384, 1280),
    show_drive_assist=True,
    max_frames=None
):
    """
    동영상 처리 및 객체 탐지
    
    Args:
        video_path: 입력 동영상 경로
        output_path: 출력 동영상 경로
        checkpoint_path: 모델 체크포인트
        img_size: 모델 입력 크기
        show_drive_assist: 자율주행 보조 표시
        max_frames: 최대 프레임 수 (None이면 전체)
    """
    class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 
                  'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    
    # 모델 로드
    model, decoder, device = load_model(checkpoint_path)
    
    # 동영상 열기
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"동영상을 열 수 없습니다: {video_path}")
        return
    
    # 동영상 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"   해상도: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   총 프레임: {total_frames}")
    
    # 출력 동영상 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break
            
            frame_count += 1
            
            # 객체 탐지
            boxes, scores, classes, scales = process_frame(
                frame, model, decoder, device, img_size
            )
            
            # 결과 그리기
            output_frame, stats = draw_detections(
                frame, boxes, scores, classes, scales, class_names
            )
            
            # 자율주행 보조 표시
            if show_drive_assist:
                should_stop, reason = check_stop_condition(
                    boxes, scores, classes, scales, class_names
                )
                
                # 상태 표시
                status_text = "STOP" if should_stop else "GO"
                status_color = (0, 0, 255) if should_stop else (0, 255, 0)
                
                cv2.rectangle(output_frame, (10, 10), (150, 70), (0, 0, 0), -1)
                cv2.putText(output_frame, status_text, (20, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
                
                if should_stop:
                    cv2.putText(output_frame, reason, (20, height - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 프레임 정보
            info_text = f"Frame: {frame_count}/{total_frames} | Objects: {stats['total']}"
            cv2.putText(output_frame, info_text, (width - 400, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 저장
            out.write(output_frame)
            
            # 진행률
            if frame_count % 30 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100
                print(f"  [{progress:.1f}%] {frame_count}/{total_frames} frames")
        
    finally:
        cap.release()
        out.release()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='동영상 객체 탐지')
    parser.add_argument('--video', type=str, default='video.mp4',
                       help='입력 동영상 경로')
    parser.add_argument('--output', type=str, default='output.mp4',
                       help='출력 동영상 경로')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                       help='모델 체크포인트')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='최대 프레임 수 (테스트용)')
    parser.add_argument('--no_assist', action='store_true',
                       help='자율주행 보조 끄기')
    
    args = parser.parse_args()
    
    process_video(
        args.video,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        show_drive_assist=not args.no_assist,
        max_frames=args.max_frames
    )
