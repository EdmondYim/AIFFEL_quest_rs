"""
KITTI 데이터셋 로더
객체 탐지를 위한 KITTI 데이터셋을 로드하고 파싱합니다.
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


class KITTIDataset(Dataset):
    """
    KITTI 객체 탐지 데이터셋
    
    Args:
        img_dir: 이미지 디렉토리 경로
        label_dir: 레이블 디렉토리 경로
        classes: 사용할 클래스 리스트 (None이면 모든 클래스 사용)
        exclude_dontcare: DontCare 객체 제외 여부
    """
    
    # KITTI 데이터셋의 모든 클래스
    ALL_CLASSES = [
        'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 
        'Cyclist', 'Tram', 'Misc', 'DontCare'
    ]
    
    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        classes: Optional[List[str]] = None,
        exclude_dontcare: bool = True
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.exclude_dontcare = exclude_dontcare
        
        # 사용할 클래스 설정
        if classes is None:
            self.classes = [c for c in self.ALL_CLASSES if c != 'DontCare']
        else:
            self.classes = classes
            
        # 클래스명 -> 인덱스 매핑 (0부터 시작)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # 이미지 파일 리스트 가져오기
        self.image_files = self._get_image_files()
        
        print(f"KITTIDataset 초기화 완료:")
        print(f"  - 이미지 개수: {len(self.image_files)}")
        print(f"  - 클래스 개수: {len(self.classes)}")
        print(f"  - 클래스: {self.classes}")
        
    def _get_image_files(self) -> List[str]:
        """이미지 파일 목록을 가져오고 검증합니다."""
        image_files = []
        
        # 이미지 디렉토리의 모든 PNG 파일 찾기
        for fname in sorted(os.listdir(self.img_dir)):
            if fname.endswith('.png') or fname.endswith('.jpg'):
                img_path = os.path.join(self.img_dir, fname)
                label_path = os.path.join(self.label_dir, fname.replace('.png', '.txt').replace('.jpg', '.txt'))
                
                # 이미지와 레이블 모두 존재하는지 확인
                if os.path.exists(img_path) and os.path.exists(label_path):
                    image_files.append(fname)
                else:
                    print(f"경고: {fname}의 레이블 파일이 없습니다.")
                    
        return image_files
    
    def _parse_label(self, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        레이블 파일을 파싱하여 bbox와 클래스 정보를 추출합니다.
        
        Returns:
            boxes: (N, 4) numpy array of [x_min, y_min, x_max, y_max]
            labels: (N,) numpy array of class indices
        """
        boxes = []
        labels = []
        
        if not os.path.exists(label_path):
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) < 15:
                    continue
                
                class_name = parts[0]
                
                # DontCare 제외 옵션 처리
                if self.exclude_dontcare and class_name == 'DontCare':
                    continue
                
                # 사용할 클래스에 포함되지 않으면 스킵
                if class_name not in self.class_to_idx:
                    continue
                
                # Bounding box 좌표 (2D bbox)
                try:
                    bbox = [float(parts[4]), float(parts[5]), 
                           float(parts[6]), float(parts[7])]
                    
                    # 유효한 bbox인지 확인
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        boxes.append(bbox)
                        labels.append(self.class_to_idx[class_name])
                except (ValueError, IndexError):
                    continue
        
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    def __len__(self) -> int:
        """데이터셋의 크기를 반환합니다."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        인덱스에 해당하는 샘플을 반환합니다.
        
        Returns:
            dict: {
                'image': PIL.Image,
                'boxes': np.ndarray (N, 4),
                'labels': np.ndarray (N,),
                'image_id': str
            }
        """
        # 파일명 가져오기
        fname = self.image_files[idx]
        img_path = os.path.join(self.img_dir, fname)
        label_path = os.path.join(self.label_dir, fname.replace('.png', '.txt').replace('.jpg', '.txt'))
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        
        # 레이블 파싱
        boxes, labels = self._parse_label(label_path)
        
        # 결과 딕셔너리 생성
        target = {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': fname.replace('.png', '').replace('.jpg', '')
        }
        
        return target
    
    def get_class_name(self, idx: int) -> str:
        """클래스 인덱스를 클래스 이름으로 변환합니다."""
        return self.idx_to_class.get(idx, 'Unknown')
    
    def get_num_classes(self) -> int:
        """클래스 개수를 반환합니다 (배경 제외)."""
        return len(self.classes)


def test_dataset(num_samples=10):
    """
    데이터셋 기본 기능 테스트 (빠른 검증)
    
    Args:
        num_samples: 통계 계산에 사용할 샘플 수 (전체 데이터셋 순회 방지)
    """
    print("=" * 50)
    print("KITTI Dataset 테스트")
    print("=" * 50)
    
    # 데이터셋 초기화
    dataset = KITTIDataset(
        img_dir='archive/data_object_image_2/training/image_2',
        label_dir='archive/data_object_label_2/training/label_2'
    )
    
    print(f"\n데이터셋 크기: {len(dataset)}")
    print(f"클래스 개수: {dataset.get_num_classes()}")
    print(f"클래스 목록: {dataset.classes}")
    
    # 첫 번째 샘플 확인
    print("\n첫 번째 샘플:")
    sample = dataset[0]
    print(f"  - Image ID: {sample['image_id']}")
    print(f"  - Image size: {sample['image'].size}")
    print(f"  - Boxes shape: {sample['boxes'].shape}")
    print(f"  - Labels shape: {sample['labels'].shape}")
    print(f"  - Number of objects: {len(sample['labels'])}")
    
    if len(sample['labels']) > 0:
        print(f"\n첫 번째 객체 정보:")
        print(f"  - Class: {dataset.get_class_name(sample['labels'][0])}")
        print(f"  - Bbox: {sample['boxes'][0]}")
    
    # 통계 정보 (일부 샘플만 사용)
    print(f"\n데이터셋 통계 (처음 {num_samples}개 샘플 기준):")
    total_objects = 0
    class_counts = {cls: 0 for cls in dataset.classes}
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        total_objects += len(sample['labels'])
        
        for label in sample['labels']:
            class_name = dataset.get_class_name(label)
            if class_name in class_counts:
                class_counts[class_name] += 1
    
    print(f"  - 총 객체 수: {total_objects}")
    print(f"  - 평균 객체 수/이미지: {total_objects / min(num_samples, len(dataset)):.2f}")
    print(f"\n클래스별 객체 수:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  - {cls}: {count}")
    
    print("\n" + "=" * 50)
    print("테스트 완료!")
    print("=" * 50)


if __name__ == '__main__':
    test_dataset()
