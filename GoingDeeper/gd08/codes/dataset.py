from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import io
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from annotation import parse_one_annotation
from utils import crop_roi, make_heatmaps

class MPIIDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # JSON 파일을 읽어 annotations 리스트 생성
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        # 각 annotation을 파싱하여 리스트에 저장
        self.annotations = [parse_one_annotation(anno, image_dir) for anno in annotations]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        # 이미지 파일 경로로부터 이미지를 로드 (RGB 모드로 변환)
        image = Image.open(anno['filepath']).convert('RGB')

        # transform이 있으면 적용
        if self.transform:
            image, heatmaps = self.transform({'image': image, 'annotation': anno})
            return image, heatmaps
        else:
            # transform이 없으면 원본 이미지와 annotation dict 반환
            return image, anno

class Preprocessor(object):
    def __init__(self,
                 image_shape=(256, 256, 3),
                 heatmap_shape=(64, 64, 16),
                 is_train=False):
        self.is_train = is_train
        self.image_shape = image_shape      # (height, width, channels)
        self.heatmap_shape = heatmap_shape  # (height, width, num_heatmap)

    def __call__(self, example):
        features = self.parse_tfexample(example)
        # image 데이터를 직접 사용 (불필요한 인코딩/디코딩 제거)
        image = example['image']

        if self.is_train:
            # 0.1 ~ 0.3 사이의 random margin 생성
            random_margin = torch.empty(1).uniform_(0.1, 0.3).item()
            image, keypoint_x, keypoint_y = crop_roi(image, features, margin=random_margin)
            image = image.resize((self.image_shape[1], self.image_shape[0]))
        else:
            image, keypoint_x, keypoint_y = crop_roi(image, features)
            image = image.resize((self.image_shape[1], self.image_shape[0]))

        # 이미지 정규화: uint8 → [0,255] → [-1, 1]
        image_np = np.array(image).astype(np.float32)
        image_np = image_np / 127.5 - 1.0
        # 채널 우선순서로 변환: (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        heatmaps = make_heatmaps(features, keypoint_x, keypoint_y, self.heatmap_shape)

        return image_tensor, heatmaps

    def parse_tfexample(self, example):
        """
        MPIIDataset에서 전달한 예제를 받아, Preprocessor가 처리할 수 있도록 features dict를 구성합니다.
        예제 형식: {'image': PIL.Image, 'annotation': anno}
        """
        annotation = example['annotation']
        # joints: list of [x, y]
        joints = annotation['joints']
        keypoint_x = [joint[0] for joint in joints]
        keypoint_y = [joint[1] for joint in joints]

        # joints_vis가 없으면 모든 관절이 가시적이라고 가정 (1)
        joints_vis = annotation.get('joints_vis', [1] * len(joints))

        features = {
            # 'image/encoded': self.image_to_bytes(example['image']), # 제거됨
            'image/object/parts/x': keypoint_x,
            'image/object/parts/y': keypoint_y,
            'image/object/parts/v': joints_vis,
            'image/object/center/x': annotation['center'][0],
            'image/object/center/y': annotation['center'][1],
            'image/object/scale': annotation['scale'],
        }
        return features



IMAGE_SHAPE = (256, 256, 3)
HEATMAP_SIZE = (64, 64)

def create_dataloader(annotation_file, image_dir, batch_size, num_heatmap, is_train=True):
    """
    annotation_file: JSON 파일 경로 (예: train.json)
    image_dir: 이미지 파일들이 저장된 디렉토리 경로
    batch_size: 배치 크기
    num_heatmap: 생성할 heatmap 개수
    is_train: True이면 shuffle 적용
    """

    preprocess = Preprocessor(
        image_shape=IMAGE_SHAPE,
        heatmap_shape=(HEATMAP_SIZE[0], HEATMAP_SIZE[1], num_heatmap),
        is_train=is_train
    )

    dataset = MPIIDataset(annotation_file=annotation_file, image_dir=image_dir, transform=preprocess)

    # DataLoader 생성
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=None
    )

    return dataloader

def show_dataset_samples(json_file, image_dir):
    # Transform 없이 원본 데이터셋 로드
    dataset = MPIIDataset(annotation_file=json_file, image_dir=image_dir, transform=None)

    # 데이터셋 크기 확인
    total_len = len(dataset)
    print(f"Dataset size: {total_len}")

    # 랜덤하게 9개 인덱스 선택
    if total_len < 9:
        sample_indices = range(total_len)
    else:
        sample_indices = random.sample(range(total_len), 9)

    # 3x3 그리드 생성
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i, idx in enumerate(sample_indices):
        image, anno = dataset[idx]

        # 이미지 표시
        axes[i].imshow(image)
        axes[i].set_title(f"Index: {idx}\n{anno['filename']}")
        axes[i].axis('off')

        # Keypoints (joints) 표시
        joints = anno['joints']
        visibility = anno['joints_visibility']

        # x, y 좌표 분리 (Visible한 것만 필터링)
        jx = []
        jy = []
        for j, v in zip(joints, visibility):
            # v가 0이면 invisible, j가 [0, 0]이거나 음수면 invalid로 간주
            if v > 0 and j[0] > 0 and j[1] > 0:
                jx.append(j[0])
                jy.append(j[1])

        # 점 찍기 (빨간색)
        axes[i].scatter(jx, jy, s=20, c='red', marker='o', label='Keypoints')

    plt.tight_layout()
    plt.show()
