import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from face_detector.model import SSD
from face_detector.dataset import WiderFaceDataset, default_box
from inference import parse_predict

def letterbox_resize(img, target_size):
    """
    종횡비를 유지하면서 레터박스 패딩으로 이미지 크기 조정.
    Args:
        img: (H, W, C) numpy 배열
        target_size: (target_h, target_w) 튜플
    Returns:
        padded_img: (target_h, target_w, C) numpy 배열
        ratio: float, 리사이즈 비율
        pad: (pad_w, pad_h) 튜플
    """
    target_h, target_w = target_size
    h, w = img.shape[:2]
    
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    
    resized_img = cv2.resize(img, (nw, nh))
    
    pad_w = (target_w - nw) // 2
    pad_h = (target_h - nh) // 2
    
    padded_img = np.full((target_h, target_w, 3), 128, dtype=np.uint8) # 회색 패딩
    padded_img[pad_h:pad_h+nh, pad_w:pad_w+nw] = resized_img
    
    return padded_img, scale, (pad_w, pad_h)

class BenchmarkDataset(WiderFaceDataset):
    def __init__(self, root_path, split='val', input_size=(256, 320)):
        # 커스텀 전처리를 구현하므로 transform 없이 부모 클래스 초기화
        super().__init__(root_path, split=split, transform=None)
        self.input_size = input_size # (H, W)
        
    def __getitem__(self, idx):
        filename, raw_boxes = self.infos[idx]
        image_path = os.path.join(self.image_dir, filename)
        
        # 이미지 로드
        img = cv2.imread(image_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 레터박스 리사이즈
        padded_img, scale, (pad_w, pad_h) = letterbox_resize(img, self.input_size)
        
        # 정규화
        padded_img = padded_img.astype(np.float32) / 255.0
        # Mean/Std 정규화 (ImageNet) - 학습 시 사용하지 않았으므로 제거
        # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # padded_img = (padded_img - mean) / std
        
        # 텐서로 변환
        img_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).float()
        
        # Ground Truth 박스 처리
        # raw_boxes: [x, y, w, h]
        gt_boxes = []
        for box in raw_boxes:
            x, y, w, h = box
            # 평가를 위해 원본 박스 유지 (예측을 원본 좌표로 매핑하면 GT 리사이즈 불필요)
            # 하지만 mAP 계산을 위해서는 예측을 원본 이미지 좌표로 매핑하는 것이 더 쉬움
            gt_boxes.append([x, y, x+w, y+h]) # [xmin, ymin, xmax, ymax]
            
        return img_tensor, np.array(gt_boxes), (img.shape[0], img.shape[1]), scale, (pad_w, pad_h)

def calculate_iou(box1, box2):
    """
    두 박스 집합 간의 IoU 계산.
    box1: (N, 4) [xmin, ymin, xmax, ymax]
    box2: (M, 4) [xmin, ymin, xmax, ymax]
    Returns: (N, M) IoU 행렬
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    lt = np.maximum(box1[:, None, :2], box2[:, :2])
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])
    
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    return inter / np.clip(union, 1e-6, None)

def compute_ap(recall, precision):
    """ Recall과 Precision 곡선이 주어졌을 때 Average Precision 계산 """
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate_model(model, dataset, device, score_thr, nms_thr, num_samples=None):
    model.eval()
    
    # Default box 사전 계산
    boxes = default_box(dataset.input_size[0], dataset.input_size[1]).to(device)
    
    results = []
    
    indices = range(len(dataset))
    if num_samples:
        indices = indices[:num_samples]
        
    for idx in tqdm(indices, desc="Evaluating"):
        img_tensor, gt_boxes, original_shape, scale, (pad_w, pad_h) = dataset[idx]
        
        if len(gt_boxes) == 0:
            continue
            
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(img_tensor)
            
        # 예측 결과 파싱
        pred_boxes, pred_labels, pred_scores = parse_predict(
            predictions, boxes, score_threshold=score_thr, nms_threshold=nms_thr
        )
        
        # 좌표를 원본 이미지 공간으로 복구
        if len(pred_boxes) > 0:
            # 1. 패딩 제거
            pred_boxes[:, [0, 2]] = pred_boxes[:, [0, 2]] * dataset.input_size[1] - pad_w
            pred_boxes[:, [1, 3]] = pred_boxes[:, [1, 3]] * dataset.input_size[0] - pad_h
            
            # 2. 스케일링 복원
            pred_boxes /= scale
            
            # 3. 이미지 경계에 맞춰 클리핑
            h, w = original_shape
            pred_boxes[:, [0, 2]] = np.clip(pred_boxes[:, [0, 2]], 0, w)
            pred_boxes[:, [1, 3]] = np.clip(pred_boxes[:, [1, 3]], 0, h)
            
            results.append({
                'pred_boxes': pred_boxes,
                'pred_scores': pred_scores,
                'gt_boxes': gt_boxes
            })
        else:
             results.append({
                'pred_boxes': np.array([]),
                'pred_scores': np.array([]),
                'gt_boxes': gt_boxes
            })
            
    return results

def calculate_metrics(results, iou_threshold=0.5):
    # 모든 검출과 GT를 평탄화
    all_detections = [] # (score, is_tp, difficulty, size_cat)
    total_gt = {
        'all': 0, 'easy': 0, 'medium': 0, 'hard': 0,
        'small': 0, 'med_size': 0, 'large': 0
    }
    
    for res in results:
        pred_boxes = res['pred_boxes']
        pred_scores = res['pred_scores']
        gt_boxes = res['gt_boxes']
        
        # GT 박스 분류
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        gt_difficulty = []
        gt_size_cat = []
        
        for box in gt_boxes:
            w, h = box[2] - box[0], box[3] - box[1]
            # 난이도
            if h > 50: diff = 'easy'
            elif h > 25: diff = 'medium'
            else: diff = 'hard'
            gt_difficulty.append(diff)
            total_gt[diff] += 1
            total_gt['all'] += 1
            
            # 크기
            shorter = min(w, h)
            if shorter < 32: size = 'small'
            elif shorter < 96: size = 'med_size'
            else: size = 'large'
            gt_size_cat.append(size)
            total_gt[size] += 1
            
        if len(pred_boxes) == 0:
            continue
            
        # 점수로 예측 정렬
        sort_idx = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[sort_idx]
        pred_scores = pred_scores[sort_idx]
        
        iou_matrix = calculate_iou(pred_boxes, gt_boxes)
        
        for i, pred_box in enumerate(pred_boxes):
            score = pred_scores[i]
            best_iou = 0
            best_gt_idx = -1
            
            if len(gt_boxes) > 0:
                best_gt_idx = np.argmax(iou_matrix[i])
                best_iou = iou_matrix[i, best_gt_idx]
            
            is_tp = False
            matched_diff = 'none'
            matched_size = 'none'
            
            if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                is_tp = True
                gt_matched[best_gt_idx] = True
                matched_diff = gt_difficulty[best_gt_idx]
                matched_size = gt_size_cat[best_gt_idx]
            
            all_detections.append({
                'score': score,
                'is_tp': is_tp,
                'diff': matched_diff,
                'size': matched_size
            })
            
    # 계산의 편의를 위해 DataFrame으로 변환
    df = pd.DataFrame(all_detections)
    if df.empty:
        return {k: 0.0 for k in total_gt.keys()}
        
    df = df.sort_values('score', ascending=False)
    
    metrics = {}
    
    # 각 카테고리별 AP 계산
    categories = ['all', 'easy', 'medium', 'hard', 'small', 'med_size', 'large']
    
    for cat in categories:
        if total_gt[cat] == 0:
            metrics[cat] = 0.0
            continue
            
        if cat == 'all':
            sub_df = df
        elif cat in ['easy', 'medium', 'hard']:
            # 난이도의 경우, TP는 해당 난이도와 일치할 때만 고려
            # 하지만 FP는 모든 매칭되지 않은 검출을 카운트? 
            # 표준 PASCAL VOC/COCO 방식은 여기서 복잡함.
            # 단순화된 접근: 카테고리별로 TP 필터링. 
            # 참고: 이것은 단순화된 근사치. 엄격한 WIDER FACE 평가는 더 복잡함 (어려운 얼굴 무시).
            # 여기서는 TP가 이 카테고리의 GT와 매칭되었는지만 확인.
            sub_df = df[df['diff'] == cat].copy()
            # 높은 점수의 FP가 다른 카테고리의 검출일 수 있어 까다로움.
            # 가능하면 더 간단한 클래스별 AP 계산을 사용하거나, 'is_tp' 플래그를 올바르게 사용.
            # 실제로, 표준 관행:
            # TP: 이 클래스의 GT와 매칭됨
            # FP: 매칭되지 않음 또는 다른 클래스의 GT와 매칭됨 (별도의 클래스로 취급하는 경우)
            # 하지만 여기서는 "얼굴"이라는 하나의 클래스만 있음.
            # 따라서 GT를 카테고리별로 필터링하고 그것들에 대해서만 평가해야 함.
            pass
        
        # 서브셋별 평가를 올바르게 재구현
        # 엄격하게 정확하려면 각 서브셋에 대해 매칭을 다시 실행해야 함, 
        # 서브셋에 없는 GT는 무시.
        # 하지만 그것은 비용이 많이 듦.
        # 대안: 저장한 'diff'와 'size' 태그 사용.
        
        # 서브셋 AP에 대한 올바른 접근:
        # TP: 이 서브셋의 GT와 매칭된 검출
        # FP: 어떤 GT와도 매칭되지 않은 검출 (또는 서브셋에 없는 GT와 매칭? 보통 무시됨)
        # Total Positives: 이 서브셋의 GT 개수
        
        # 저장된 'is_tp'와 카테고리 태그를 사용.
        # is_tp가 True이고 cat이 일치 -> 이 서브셋의 TP
        # is_tp가 True이고 cat이 불일치 -> 무시 (FP로 카운트하지 않고, TP로도 카운트하지 않음)
        # is_tp가 False -> 이 서브셋의 FP
        
        tp_mask = (df['is_tp'] == True) & ((df['diff'] == cat) | (df['size'] == cat) | (cat == 'all'))
        ignore_mask = (df['is_tp'] == True) & ~((df['diff'] == cat) | (df['size'] == cat) | (cat == 'all'))
        
        # Precision/Recall 계산 목표
        # TP와 FP의 누적합
        # FP는 !is_tp. 하지만 다른 카테고리와 매칭되었다면 이 카테고리의 AP에서는 무시해야 함?
        # WIDER FACE 평가 스크립트는 "무시" 영역이나 다른 클래스와 매칭된 검출을 무시함.
        # 여기서는 다음과 같이 가정:
        # TP = 특정 카테고리와 매칭됨
        # FP = 매칭되지 않음 (is_tp=False)
        # 무시됨 = 다른 카테고리와 매칭됨
        
        subset_df = df.copy()
        subset_df['this_tp'] = tp_mask
        subset_df['ignored'] = ignore_mask
        
        # 무시된 검출 필터링
        subset_df = subset_df[~subset_df['ignored']]
        
        tp = subset_df['this_tp'].cumsum()
        fp = (~subset_df['this_tp']).cumsum()
        
        rec = tp / total_gt[cat]
        prec = tp / (tp + fp + 1e-6)
        
        ap = compute_ap(rec.values, prec.values)
        metrics[cat] = ap
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description='SSD 벤치마크')
    parser.add_argument('--model_paths', type=str, required=True, help='쉼표로 구분된 .pth 파일 경로')
    parser.add_argument('--model_names', type=str, default=None, help='쉼표로 구분된 모델 이름')
    parser.add_argument('--dataset_path', type=str, required=True, help='WIDER FACE 루트 경로')
    parser.add_argument('--num_samples', type=int, default=None, help='평가할 샘플 수')
    parser.add_argument('--input_size', type=str, default="256,320", help='H,W')
    parser.add_argument('--score_thr', type=float, default=0.8)
    parser.add_argument('--nms_thr', type=float, default=0.45)
    parser.add_argument('--output_dir', type=str, default='test_results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    model_paths = args.model_paths.split(',')
    if args.model_names:
        model_names = args.model_names.split(',')
    else:
        model_names = [os.path.basename(p) for p in model_paths]
        
    input_h, input_w = map(int, args.input_size.split(','))
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # 데이터셋 로드
    print(f"데이터셋 로딩 중: {args.dataset_path}...", flush=True)
    dataset = BenchmarkDataset(args.dataset_path, split='val', input_size=(input_h, input_w))
    print(f"데이터셋 로드 완료. 총 이미지 수: {len(dataset)}", flush=True)
    
    all_results = {}
    summary_data = []
    
    for path, name in zip(model_paths, model_names):
        print(f"\n{name} 평가 중 ({path})...", flush=True)
        
        # 모델 로드
        model = SSD(num_classes=2, input_shape=(3, input_h, input_w))
        checkpoint = torch.load(path, map_location=args.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(args.device)
        
        # 추론 실행
        results = evaluate_model(model, dataset, args.device, args.score_thr, args.nms_thr, args.num_samples)
        
        # 메트릭 계산
        metrics = calculate_metrics(results)
        
        print(f"{name} 결과:")
        print(f"  mAP@0.5: {metrics['all']:.4f}")
        print(f"  쉬움: {metrics['easy']:.4f}, 보통: {metrics['medium']:.4f}, 어려움: {metrics['hard']:.4f}")
        print(f"  작음: {metrics['small']:.4f}, 중간(크기): {metrics['med_size']:.4f}, 큼: {metrics['large']:.4f}")
        
        all_results[name] = metrics
        
        summary_row = {'Model': name}
        summary_row.update(metrics)
        summary_data.append(summary_row)
        
    # 결과 저장
    df_summary = pd.DataFrame(summary_data)
    csv_path = os.path.join(args.output_dir, 'benchmark_results.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"\n요약 저장 완료: {csv_path}")
    
    json_path = os.path.join(args.output_dir, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    # 그래프 생성
    metrics_to_plot = ['all', 'easy', 'medium', 'hard', 'small', 'med_size', 'large']
    labels = ['mAP', 'Easy', 'Medium', 'Hard', 'Small', 'Med(Size)', 'Large']
    
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(model_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, name in enumerate(model_names):
        vals = [all_results[name][m] for m in metrics_to_plot]
        ax.bar(x + i*width - 0.4 + width/2, vals, width, label=name)
        
    ax.set_ylabel('AP')
    ax.set_title('모델 및 카테고리별 벤치마크 결과')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'benchmark_chart.png'))
    print("차트 저장 완료.")

if __name__ == '__main__':
    main()
