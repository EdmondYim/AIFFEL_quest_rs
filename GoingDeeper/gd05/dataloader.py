"""
KITTI 데이터로더 및 데이터 증강
객체 탐지를 위한 데이터 증강 및 DataLoader 생성 함수를 제공합니다.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import random
from PIL import Image
from typing import Dict, List, Tuple, Optional, Callable
from dataset import KITTIDataset


class Compose:
    """여러 변환을 순차적으로 적용합니다."""
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """PIL Image를 Tensor로 변환합니다."""
    
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        return image, target


class Normalize:
    """이미지를 정규화합니다 (ImageNet 평균/표준편차)."""
    
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target):
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomHorizontalFlip:
    """랜덤 좌우 반전 + bbox 좌표 조정."""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, target):
        if random.random() < self.p:
            # 이미지가 PIL인지 Tensor인지 확인
            if isinstance(image, Image.Image):
                image = TF.hflip(image)
                width, _ = image.size
            else:  # Tensor
                image = TF.hflip(image)
                _, _, width = image.shape
            
            # Bounding box 좌표 변환
            boxes = target['boxes'].copy()
            if len(boxes) > 0:
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target['boxes'] = boxes
        
        return image, target


class Resize:
    """고정 크기로 리사이즈 + bbox 스케일링."""
    
    def __init__(self, size: Tuple[int, int]):
        """
        Args:
            size: (height, width) tuple
        """
        self.size = size  # (H, W)
    
    def __call__(self, image, target):
        # 원본 크기
        if isinstance(image, Image.Image):
            orig_width, orig_height = image.size
            image = TF.resize(image, self.size)
        else:  # Tensor
            _, orig_height, orig_width = image.shape
            image = TF.resize(image, self.size)
        
        # 스케일 계산
        new_height, new_width = self.size
        scale_y = new_height / orig_height
        scale_x = new_width / orig_width
        
        # Bounding box 스케일링
        boxes = target['boxes'].copy()
        if len(boxes) > 0:
            boxes[:, 0] *= scale_x  # x_min
            boxes[:, 1] *= scale_y  # y_min
            boxes[:, 2] *= scale_x  # x_max
            boxes[:, 3] *= scale_y  # y_max
            target['boxes'] = boxes
        
        return image, target

'''
class RandomResize:
    """랜덤 크기 조정 (aspect ratio 유지)."""
    
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size
    
    def __call__(self, image, target):
        # 원본 크기
        if isinstance(image, Image.Image):
            orig_width, orig_height = image.size
        else:
            _, orig_height, orig_width = image.shape
        
        # 랜덤 크기 선택
        size = random.randint(self.min_size, self.max_size)
        
        # Aspect ratio 유지하며 리사이즈
        if orig_height < orig_width:
            new_height = size
            new_width = int(size * orig_width / orig_height)
        else:
            new_width = size
            new_height = int(size * orig_height / orig_width)
        
        # 최대 크기 제한
        if max(new_height, new_width) > self.max_size:
            if new_height > new_width:
                new_height = self.max_size
                new_width = int(self.max_size * orig_width / orig_height)
            else:
                new_width = self.max_size
                new_height = int(self.max_size * orig_height / orig_width)
        
        # 리사이즈 적용
        resize_transform = Resize((new_height, new_width))
        return resize_transform(image, target)
'''

class ColorJitter:
    """색상 변환: 밝기, 대비, 채도, 색조 조정."""
    
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, image, target):
        # PIL Image에만 적용 (Tensor 이전에 적용되어야 함)
        if isinstance(image, Image.Image):
            if self.brightness > 0:
                brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
                image = TF.adjust_brightness(image, brightness_factor)
            
            if self.contrast > 0:
                contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
                image = TF.adjust_contrast(image, contrast_factor)
            
            if self.saturation > 0:
                saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
                image = TF.adjust_saturation(image, saturation_factor)
            
            if self.hue > 0:
                hue_factor = random.uniform(-self.hue, self.hue)
                image = TF.adjust_hue(image, hue_factor)
        
        return image, target


class RandomCrop:
    """랜덤 크롭 + bbox 클리핑 및 유효성 검증."""
    
    def __init__(self, size: Tuple[int, int], min_area_ratio=0.3):
        """
        Args:
            size: (height, width) 크롭 크기
            min_area_ratio: 크롭 후 최소 면적 비율 (원본 대비)
        """
        self.size = size
        self.min_area_ratio = min_area_ratio
    
    def __call__(self, image, target):
        # 원본 크기
        if isinstance(image, Image.Image):
            orig_width, orig_height = image.size
        else:
            _, orig_height, orig_width = image.shape
        
        crop_height, crop_width = self.size
        
        # 이미지가 크롭 크기보다 작으면 패딩
        if orig_height < crop_height or orig_width < crop_width:
            # 패딩 없이 리사이즈로 처리
            resize_transform = Resize(self.size)
            return resize_transform(image, target)
        
        # 랜덤 크롭 위치
        top = random.randint(0, orig_height - crop_height)
        left = random.randint(0, orig_width - crop_width)
        
        # 이미지 크롭
        if isinstance(image, Image.Image):
            image = TF.crop(image, top, left, crop_height, crop_width)
        else:
            image = TF.crop(image, top, left, crop_height, crop_width)
        
        # Bounding box 조정
        boxes = target['boxes'].copy()
        labels = target['labels'].copy()
        
        if len(boxes) > 0:
            # 크롭 영역으로 좌표 변환
            boxes[:, 0] -= left  # x_min
            boxes[:, 1] -= top   # y_min
            boxes[:, 2] -= left  # x_max
            boxes[:, 3] -= top   # y_max
            
            # 크롭 영역 내로 클리핑
            boxes[:, 0] = np.clip(boxes[:, 0], 0, crop_width)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, crop_height)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, crop_width)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, crop_height)
            
            # 유효한 bbox만 유지 (최소 면적 체크)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            orig_areas = (target['boxes'][:, 2] - target['boxes'][:, 0]) * \
                        (target['boxes'][:, 3] - target['boxes'][:, 1])
            
            valid_mask = (boxes[:, 2] > boxes[:, 0]) & \
                        (boxes[:, 3] > boxes[:, 1]) & \
                        (areas >= orig_areas * self.min_area_ratio)
            
            boxes = boxes[valid_mask]
            labels = labels[valid_mask]
        
        target['boxes'] = boxes
        target['labels'] = labels
        
        return image, target


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    배치 생성을 위한 커스텀 collate 함수.
    이미지 크기가 다를 수 있으므로 리스트로 반환합니다.
    
    Args:
        batch: 샘플 리스트
    
    Returns:
        images: (B, C, H, W) Tensor
        targets: 타겟 딕셔너리 리스트
    """
    images = []
    targets = []
    
    for item in batch:
        images.append(item['image'])
        targets.append({
            'boxes': torch.tensor(item['boxes'], dtype=torch.float32),
            'labels': torch.tensor(item['labels'], dtype=torch.int64),
            'image_id': item['image_id']
        })
    
    # 이미지를 스택 (모두 같은 크기여야 함)
    images = torch.stack(images, dim=0)
    
    return images, targets


class TransformedDataset:
    """Transform을 적용하는 데이터셋 래퍼."""
    
    def __init__(self, dataset: KITTIDataset, transform: Optional[Callable] = None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        target = {
            'boxes': sample['boxes'],
            'labels': sample['labels'],
            'image_id': sample['image_id']
        }
        
        if self.transform is not None:
            image, target = self.transform(image, target)
        
        return {
            'image': image,
            'boxes': target['boxes'],
            'labels': target['labels'],
            'image_id': target['image_id']
        }


def get_train_transforms(img_size=(600, 800)):
    """훈련용 데이터 변환 파이프라인."""
    return Compose([
        # RandomResize(min_size=580, max_size=620), 필요없는 증강.
        RandomHorizontalFlip(p=0.3),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        Resize(img_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(img_size=(600, 800)):
    """검증용 데이터 변환 파이프라인 (증강 없음)."""
    return Compose([
        Resize(img_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_train_dataloader(
    img_dir: str,
    label_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (600, 800),
    shuffle: bool = True,
    classes: Optional[List[str]] = None
) -> DataLoader:
    """
    훈련용 DataLoader를 생성합니다.
    
    Args:
        img_dir: 이미지 디렉토리
        label_dir: 레이블 디렉토리
        batch_size: 배치 크기
        num_workers: 워커 프로세스 수
        img_size: 이미지 크기 (height, width)
        shuffle: 셔플 여부
        classes: 사용할 클래스 리스트
    
    Returns:
        DataLoader
    """
    dataset = KITTIDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        classes=classes,
        exclude_dontcare=True
    )
    
    transformed_dataset = TransformedDataset(
        dataset=dataset,
        transform=get_train_transforms(img_size)
    )
    
    return DataLoader(
        transformed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def get_val_dataloader(
    img_dir: str,
    label_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (600, 800),
    shuffle: bool = False,
    classes: Optional[List[str]] = None
) -> DataLoader:
    """
    검증용 DataLoader를 생성합니다.
    
    Args:
        img_dir: 이미지 디렉토리
        label_dir: 레이블 디렉토리
        batch_size: 배치 크기
        num_workers: 워커 프로세스 수
        img_size: 이미지 크기 (height, width)
        shuffle: 셔플 여부
        classes: 사용할 클래스 리스트
    
    Returns:
        DataLoader
    """
    dataset = KITTIDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        classes=classes,
        exclude_dontcare=True
    )
    
    transformed_dataset = TransformedDataset(
        dataset=dataset,
        transform=get_val_transforms(img_size)
    )
    
    return DataLoader(
        transformed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def test_dataloader():
    """DataLoader 기본 기능 테스트"""
    print("=" * 50)
    print("KITTI DataLoader 테스트")
    print("=" * 50)
    
    # 훈련용 DataLoader
    train_loader = get_train_dataloader(
        img_dir='archive/data_object_image_2/training/image_2',
        label_dir='archive/data_object_label_2/training/label_2',
        batch_size=8,
        num_workers=4,  # 0 >> 4
        img_size=(600, 800),
        shuffle=True
    )
    
    print(f"\n훈련 DataLoader 생성 완료")
    print(f"배치 수: {len(train_loader)}")
    
    # 첫 번째 배치 확인
    images, targets = next(iter(train_loader))
    
    print(f"\n첫 번째 배치:")
    print(f"  - Images shape: {images.shape}")
    print(f"  - Number of samples: {len(targets)}")
    print(f"  - First sample boxes shape: {targets[0]['boxes'].shape}")
    print(f"  - First sample labels shape: {targets[0]['labels'].shape}")
    
    # 여러 배치 순회 테스트
    print(f"\n처음 5개 배치 순회 테스트:")
    for batch_idx, (images, targets) in enumerate(train_loader):
        print(f"  Batch {batch_idx}: images {images.shape}, {len(targets)} targets")
        if batch_idx >= 4:
            break
    
    print("\n" + "=" * 50)
    print("테스트 완료!")
    print("=" * 50)


if __name__ == '__main__':
    test_dataloader()
