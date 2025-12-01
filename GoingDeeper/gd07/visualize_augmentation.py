import os
from glob import glob
from imageio import imread
import matplotlib.pyplot as plt
from albumentations import HorizontalFlip, Compose, Resize, ColorJitter


def build_augmentation(is_train=True):
    if is_train:    # 훈련용 데이터일 경우
        return Compose([
            HorizontalFlip(p=0.5),    # 50%의 확률로 좌우대칭
            ColorJitter(              # 채도, 명도, 대비 변경 (image만 적용)
                brightness=0.2,       # 명도 ±20%
                contrast=0.2,         # 대비 ±20%
                saturation=0.2,       # 채도 ±20%
                hue=0.1,              # 색조 ±10%
                p=0.8                 # 80% 확률로 적용
            ),
            Resize(                   # 원본 비율 유지 (약 3:1), U-Net에 최적화된 크기
                width=768,
                height=256
            )
        ], additional_targets={'mask': 'mask'})  # mask도 함께 transform
    return Compose([      # 테스트용 데이터일 경우에는 768x256으로 resize만 수행합니다.
        Resize(
            width=768,
            height=256
        )
    ], additional_targets={'mask': 'mask'})


if __name__ == "__main__":
    # 데이터 디렉토리 설정 (사용자 환경에 맞게 수정 필요)
    data_dir = 'data_semantics/training'
    
    # Augmentation 빌드
    train_aug = build_augmentation(is_train=True)
    test_aug = build_augmentation(is_train=False)
    
    # 이미지 경로 가져오기
    image_paths = sorted(glob(os.path.join(data_dir, "image_2", "*.png")))
    mask_paths = sorted(glob(os.path.join(data_dir, "semantic", "*.png")))
    
    if len(image_paths) > 0:
        # 첫 번째 이미지로 테스트
        img_path = image_paths[0]
        mask_path = mask_paths[0]
        
        image = imread(img_path)
        mask = imread(mask_path)
        
        # Augmentation 적용
        data = {"image": image, "mask": mask}
        augmented = train_aug(**data)
        
        aug_image = augmented["image"]
        aug_mask = augmented["mask"]
        
        # 시각화
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        
        ax[0, 0].imshow(image)
        ax[0, 0].set_title("Original Image")
        
        ax[0, 1].imshow(mask)
        ax[0, 1].set_title("Original Mask")
        
        ax[1, 0].imshow(aug_image)
        ax[1, 0].set_title("Augmented Image")
        
        ax[1, 1].imshow(aug_mask)
        ax[1, 1].set_title("Augmented Mask")
        
        plt.tight_layout()
        plt.show()

