"""
추론 결과 시각화
RetinaNet으로 탐지된 객체를 이미지에 그려서 시각화
"""

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from torchvision import transforms

from model import RetinaNet, DecodePredictions, get_resnet50_backbone


# 클래스별 색상 매핑
CLASS_COLORS = {
    'Car': (255, 0, 0),           # 빨강
    'Van': (255, 128, 0),         # 주황
    'Truck': (255, 255, 0),       # 노랑
    'Pedestrian': (0, 255, 0),    # 초록
    'Person_sitting': (0, 255, 128),  # 연두
    'Cyclist': (0, 128, 255),     # 하늘색
    'Tram': (128, 0, 255),        # 보라
    'Misc': (255, 0, 255)         # 
}


def load_model(checkpoint_path='checkpoints/best.pth', num_classes=8, device='cuda'):
    """학습된 모델 로드"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    backbone = get_resnet50_backbone(pretrained=False)
    model = RetinaNet(num_classes=num_classes, backbone=backbone)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"모델 로드 완료: {checkpoint_path}")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    except FileNotFoundError:
        print(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")

    
    model = model.to(device)
    model.eval()
    
    decoder = DecodePredictions(
        num_classes=num_classes,
        confidence_threshold=0.3,
        nms_iou_threshold=0.5,
        max_detections=100
    )
    
    return model, decoder, device


def check_stop_condition(boxes, scores, classes, scales, class_names, size_limit=300):
    """
    정지 조건 확인
    
    조건:
    1. 사람(Pedestrian, Person_sitting, Cyclist) 탐지 시 -> Stop
    2. 차량(Car, Van, Truck) 크기 >= 300px 시 -> Stop
    
    Args:
        boxes: 바운딩 박스 배열
        scores: 점수 배열
        classes: 클래스 배열
        scales: (scale_w, scale_h)
        class_names: 클래스 이름 리스트
        size_limit: 차량 크기 제한
    
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
            return True, f"Person ({class_names[cls]})"
    
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
                return True, f"Large {class_names[cls]} ({max(width, height):.0f}px)"
    
    return False, "Safe"


def visualize_detection(
    image_path,
    checkpoint_path='checkpoints/best.pth',
    output_path=None,
    img_size=(384, 1280),
    show_scores=True,
    min_score=0.3,
    show_drive_assist=True
):
    """
    이미지에 탐지 결과 시각화
    
    Args:
        image_path: 입력 이미지 경로
        checkpoint_path: 모델 체크포인트
        output_path: 저장 경로 (None이면 화면에 표시)
        img_size: 모델 입력 크기
        show_scores: 점수 표시 여부
        show_drive_assist: 자율주행 보조 표시 여부
        min_score: 최소 표시 점수
    
    Returns:
        PIL Image with bounding boxes
    """
    # 클래스 이름
    class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 
                  'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    
    # 모델 로드
    model, decoder, device = load_model(checkpoint_path)
    
    # 원본 이미지 로드
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    # 스케일 계산
    scale_w = img_size[1] / original_size[0]
    scale_h = img_size[0] / original_size[1]
    
    # 전처리
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 추론
    with torch.no_grad():
        predictions = model(image_tensor)
        decoded = decoder(image_tensor, predictions)
    
    boxes, scores, classes = decoded[0]
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    classes = classes.cpu().numpy()
    
    # 원본 이미지에 그리기
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # 폰트 설정 (한글 지원)
    try:
        font = ImageFont.truetype("malgun.ttf", 20)  # Windows
    except:
        font = ImageFont.load_default()
    
    detection_count = 0
    class_counts = {}
    
    # 각 탐지 결과 그리기
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        if score < min_score:
            continue
        
        detection_count += 1
        
        # 원본 좌표로 변환
        x1 = box[0] / scale_w
        y1 = box[1] / scale_h
        x2 = box[2] / scale_w
        y2 = box[3] / scale_h
        
        # 클래스 정보
        class_name = class_names[cls]
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        
        # 카운트
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # 바운딩 박스 그리기
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 라벨 텍스트
        if show_scores:
            label = f"{class_name} {score:.2f}"
        else:
            label = class_name
        
        # 라벨 배경
        text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 25), label, fill=(255, 255, 255), font=font)
    
    # 자율주행 보조 표시
    if show_drive_assist:
        should_stop, reason = check_stop_condition(
            boxes, scores, classes, (scale_w, scale_h), class_names
        )
        
        # 상태 텍스트
        status_text = "STOP" if should_stop else "GO"
        status_color = (255, 0, 0) if should_stop else (0, 255, 0)  # 빨강/초록
        
        # 배경 박스 (좌측 상단)
        try:
            status_font = ImageFont.truetype("malgun.ttf", 60)
        except:
            status_font = ImageFont.load_default()
        
        # 상태 표시 박스
        draw.rectangle([(10, 10), (200, 90)], fill=(0, 0, 0))
        draw.text((20, 20), status_text, fill=status_color, font=status_font)
        
        # 이유 표시 (STOP인 경우)
        if should_stop:
            try:
                reason_font = ImageFont.truetype("malgun.ttf", 25)
            except:
                reason_font = ImageFont.load_default()
            
            draw.text((20, original_size[1] - 40), f"Reason: {reason}", 
                     fill=(255, 0, 0), font=reason_font)
    
    # 결과 출력
    print(f"\n 탐지 완료!")
    print(f"   총 {detection_count}개 객체 탐지")
    if class_counts:
        print("   클래스별 개수:")
        for cls_name, count in sorted(class_counts.items()):
            print(f"     - {cls_name}: {count}")
    
    if show_drive_assist:
        print(f"   자율주행 상태: {status_text} ({reason})")
    # 저장 또는 표시
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        draw_image.save(output_path)
        print(f"\n결과 저장: {output_path}")
    else:
        draw_image.show()
    
    return draw_image


def batch_visualize(
    image_dir,
    checkpoint_path='checkpoints/best.pth',
    output_dir='results',
    max_images=10
):
    """
    여러 이미지 배치 처리
    
    Args:
        image_dir: 이미지 디렉토리
        checkpoint_path: 모델 체크포인트
        output_dir: 결과 저장 디렉토리
        max_images: 최대 처리 이미지 수
    """
    import glob
    
    # 이미지 파일 찾기
    image_files = glob.glob(os.path.join(image_dir, '*.png'))
    image_files += glob.glob(os.path.join(image_dir, '*.jpg'))
    image_files = sorted(image_files)[:max_images]
    
    print(f"{len(image_files)}개 이미지 처리 시작...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {os.path.basename(img_path)}")
        
        output_path = os.path.join(
            output_dir, 
            f"detected_{os.path.basename(img_path)}"
        )
        
        try:
            visualize_detection(
                img_path,
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                show_scores=True
            )
        except Exception as e:
            print(f"에러: {e}")
    
    print(f"\n완료! 결과는 '{output_dir}' 폴더에 저장되었습니다.")


if __name__ == '__main__':
    import argparse
    import glob
    import random
    
    parser = argparse.ArgumentParser(description='RetinaNet 추론 시각화')
    parser.add_argument('--image', type=str, 
                       default=None,
                       help='이미지 경로 (지정하지 않으면 랜덤)')
    parser.add_argument('--image_dir', type=str,
                       default='archive/data_object_image_2/testing/image_2',
                       help='랜덤 선택할 이미지 디렉토리')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/best.pth',
                       help='모델 체크포인트')
    parser.add_argument('--output', type=str, 
                       default='result.png',
                       help='결과 저장 경로')
    parser.add_argument('--batch', action='store_true',
                       help='배치 모드')
    parser.add_argument('--batch_dir', type=str,
                       default='archive/data_object_image_2/testing/image_2',
                       help='배치 처리 디렉토리')
    parser.add_argument('--max_images', type=int, default=10,
                       help='배치 모드 최대 이미지 수')
    
    args = parser.parse_args()
    
    if args.batch:
        # 배치 처리
        batch_visualize(
            args.batch_dir,
            checkpoint_path=args.checkpoint,
            max_images=args.max_images
        )
    else:
        # result 폴더 생성
        os.makedirs('result', exist_ok=True)
        
        # 20번 반복
        for i in range(20):
            # 이미지 경로 결정
            if args.image is None:
                # 랜덤 선택
                image_files = glob.glob(os.path.join(args.image_dir, '*.png'))
                image_files += glob.glob(os.path.join(args.image_dir, '*.jpg'))
                image_path = random.choice(image_files)
            # 출력 경로 설정
            output_path = f'result/result_{i+1:02d}.png'
            # 단일 이미지 처리
            visualize_detection(
                image_path,
                checkpoint_path=args.checkpoint,
                output_path=output_path,
                show_scores=True
            )
        
     