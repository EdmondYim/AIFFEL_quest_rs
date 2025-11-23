import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import io
import math
import cv2
from .utils import encode_pt



'''
dataset.py



'''

class WiderFaceDataset(Dataset):
    def __init__(self, root_path, split='train', transform=None, boxes=None):
        self.root_path = root_path
        self.split = split
        self.transform = transform
        self.boxes = boxes
        self.infos = self._parse_widerface()
        
        # Image settings
        self.image_width = 320
        self.image_height = 256

    def _parse_box(self, data):
        x0 = int(data[0])
        y0 = int(data[1])
        w = int(data[2])
        h = int(data[3])
        return x0, y0, w, h

    def _parse_widerface(self):
        # 로컬 파일 경로
        if self.split == 'train':
            file_path = os.path.join(self.root_path, 'wider_face_split', 'wider_face_split', 'wider_face_train_bbx_gt.txt')
            self.image_dir = os.path.join(self.root_path, 'WIDER_train', 'WIDER_train', 'images')
        elif self.split == 'val':
            file_path = os.path.join(self.root_path, 'wider_face_split', 'wider_face_split', 'wider_face_val_bbx_gt.txt')
            self.image_dir = os.path.join(self.root_path, 'WIDER_val', 'WIDER_val', 'images')
        elif self.split == 'test':
            # 로컬 파일 경로
            file_path = os.path.join(self.root_path, 'wider_face_split', 'wider_face_split', 'wider_face_test_filelist.txt')
            self.image_dir = os.path.join(self.root_path, 'WIDER_test', 'WIDER_test', 'images')
            return self._parse_test_filelist(file_path)
        else:
            raise ValueError(f"스플릿 다시 하세요 {self.split}. 3 중 택 1 'train', 'val', or 'test'")

        infos = []
        if not os.path.exists(file_path):
            print(f"파일 {file_path}가 없습니다.")
            return infos

        with open(file_path) as fp:
            line = fp.readline()
            while line:
                filename = line.strip()
                if not filename: # 빈줄 헨들러
                    line = fp.readline()
                    continue
                    
                try:
                    n_object = int(fp.readline())
                except ValueError:
                    line = fp.readline()
                    continue

                boxes = []
                if n_object == 0:
                    # 박스 없으면 그대로 읽기
                    fp.readline()
                    # 박스 빈줄로 넣기
                else:
                    for i in range(n_object):
                        box_line = fp.readline().split(' ')
                        x0, y0, w, h = self._parse_box(box_line)
                        if (w == 0) or (h == 0):
                            continue
                        boxes.append([x0, y0, w, h])
                
                # 박스 있으면 넣기
                if len(boxes) > 0:
                    infos.append((filename, boxes))
                line = fp.readline()
        return infos
    
    def _parse_test_filelist(self, file_path):
        """파일 이름 파싱"""
        infos = []
        if not os.path.exists(file_path):
            print(f"경로 확인 {file_path}")
            return infos
        
        with open(file_path) as fp:
            for line in fp:
                filename = line.strip()
                if filename:
                    # 테스트 스플릿은 어노테이션 없음, 박스 빈줄로 넣기
                    infos.append((filename, []))
        
        return infos

    def _process_image(self, image_file):
        try:
            with open(image_file, 'rb') as f:
                image_string = f.read()
                image_data = Image.open(io.BytesIO(image_string)).convert('RGB')
                return image_data
        except Exception as e:
            print(f"Error reading {image_file}: {e}")
            return None

    def _crop(self, img, labels, max_loop=50):
        # 라벨도 텐서였음 ㄷㄷ (N, 5) [xmin, ymin, xmax, ymax, class]
        shape = img.shape
        
        def matrix_iof(a, b):
            lt = torch.maximum(a[:, None, :2], b[:, :2])
            rb = torch.minimum(a[:, None, 2:], b[:, 2:])

            area_i = torch.prod(rb - lt, dim=2) * (lt < rb).all(dim=2).float()
            area_a = torch.prod(a[:, 2:] - a[:, :2], dim=1)

            return area_i / torch.maximum(area_a[:, None], torch.tensor(1.0))

        for _ in range(max_loop):
            pre_scale = torch.tensor([0.3, 0.45, 0.6, 0.8, 1.0], dtype=torch.float32)
            scale = pre_scale[torch.randint(0, 5, (1,))].item()

            short_side = min(shape[1], shape[2])
            h = w = int(scale * short_side)

            if shape[1] - h + 1 <= 0 or shape[2] - w + 1 <= 0:
                continue

            h_offset = torch.randint(0, shape[1] - h + 1, (1,)).item()
            w_offset = torch.randint(0, shape[2] - w + 1, (1,)).item()

            roi = torch.tensor([w_offset, h_offset, w_offset + w, h_offset + h], dtype=torch.float32)
            
            value = matrix_iof(labels[:, :4], roi[None, :])
            if torch.any(value >= 0.7):  # 크롭 0.7 이상인 것만
                centers = (labels[:, :2] + labels[:, 2:4]) / 2

                mask_a = (roi[:2] < centers).all(dim=1) & (centers < roi[2:]).all(dim=1)
                if mask_a.any():
                    img_t = img[:, h_offset:h_offset + h, w_offset:w_offset + w]

                    labels_t = labels[mask_a]
                    labels_t[:, :4] -= torch.tensor([w_offset, h_offset, w_offset, h_offset], dtype=torch.float32)

                    return img_t, labels_t

        return img, labels

    def _resize(self, img, labels):
        # img 텐서 CHW
        h_f, w_f = img.shape[1:3]

        locs = torch.stack([labels[:, 0] / w_f, labels[:, 1] / h_f,
                            labels[:, 2] / w_f, labels[:, 3] / h_f], dim=1)

        locs = torch.clamp(locs, 0, 1.0)
        labels = torch.cat([locs, labels[:, 4].unsqueeze(1)], dim=1)

        if self.split == 'val':
            resize_case = 0 # BICUBIC
        else:
            resize_case = torch.randint(0, 3, (1,)).item()

        resize_methods = [
            T.Resize((self.image_height, self.image_width), interpolation=T.InterpolationMode.BICUBIC),
            T.Resize((self.image_height, self.image_width), interpolation=T.InterpolationMode.NEAREST),
            T.Resize((self.image_height, self.image_width), interpolation=T.InterpolationMode.BILINEAR)
        ]

        img = resize_methods[resize_case](img)

        return img, labels

    def _flip(self, img, labels):
        flip_case = torch.randint(0, 2, (1,)).item()

        if flip_case == 0:
            img = torch.flip(img, dims=[2])

            labels = torch.stack([
                1 - labels[:, 2], labels[:, 1],
                1 - labels[:, 0], labels[:, 3],
                labels[:, 4]
            ], dim=1)

        return img, labels

    def _pad_to_square(self, img, labels):
        # img 텐서 CHW, labels 텐서 (N, 5) [xmin, ymin, xmax, ymax, class]
        h, w = img.shape[1:3]
        if h == w:
            return img, labels

        dim_diff = abs(h - w)
        pad = dim_diff // 2

        if h < w:
            # 세로 패딩
            img = F.pad(img, (0, 0, pad, w - h - pad), value=img.mean())
            # Update y 좌표
            if labels.numel() > 0:
                labels[:, [1, 3]] += pad
        else:
            # 가로 패딩  
            img = F.pad(img, (pad, h - w - pad, 0, 0), value=img.mean())
            # Update x 좌표
            if labels.numel() > 0:
                labels[:, [0, 2]] += pad

        return img, labels

    def _distort(self, img):
        """강한 color augmentation"""
        # img CHW 텐서 [0, 255]
        
        if torch.rand(1).item() < 0.2:  # 20%만 스킵 (이전 40%)
            return img
        
        # 밝기 조정 (실제 조명 변동)
        if torch.rand(1).item() < 0.6:  # 60%로 증가 (이전 50%)
            delta = torch.FloatTensor(1).uniform_(-35, 35).item()  # ±35로 증가 (이전 ±25)
            img = img + delta
        
        # 사진 명도 조정
        if torch.rand(1).item() < 0.5:  # 50%로 증가 (이전 40%)
            alpha = torch.FloatTensor(1).uniform_(0.7, 1.3).item()  # 범위 확대 (이전 0.8-1.2)
            img = img * alpha
        
        # clamp
        img = torch.clamp(img, 0, 255)
        
        return img
    
    def _blur(self, img):
        """Gaussian blur 강화"""
        # img CHW 텐서 [0, 255]
        if torch.rand(1).item() < 0.25:  # 25%로 증가 (이전 20%)
            # numpy로 변환
            img_np = img.permute(1, 2, 0).numpy().astype(np.uint8)
            
            # 가우시안 블러 - 커널 사이즈 다양화
            kernel_size = torch.randint(1, 3, (1,)).item() * 2 + 1  # 3 또는 5 (이전에는 고정 3)
            
            # 가우시안 블러 적용
            img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
            
            # 텐서로 변환
            img = torch.from_numpy(img_np).permute(2, 0, 1).float()
        
        return img
    
    def _add_noise(self, img):
        """Gaussian noise 강화"""
        # img CHW 텐서 [0, 255]
        if torch.rand(1).item() < 0.25:  # 25%로 증가 (이전 15%)
            noise_factor = torch.FloatTensor(1).uniform_(5, 18).item()  # 5-18로 증가 (이전 3-12)
            noise = torch.randn_like(img) * noise_factor
            img = img + noise
            img = torch.clamp(img, 0, 255)
        
        return img
    
    def _random_erasing(self, img):
        """random erasing 강화"""
        # img CHW 텐서 [0, 255]
        if torch.rand(1).item() < 0.25:  # 25%로 증가 (이전 15%)
            h, w = img.shape[1:3]
            
            # erasing area 증가 (3% to 15% of image)
            area = h * w
            target_area = torch.FloatTensor(1).uniform_(0.03, 0.15).item() * area  # 3-15%로 증가 (이전 2-10%)
            aspect_ratio = torch.FloatTensor(1).uniform_(0.3, 3.0).item()  # 범위 확대 (이전 0.5-2.0)
            
            h_erase = int(math.sqrt(target_area * aspect_ratio))
            w_erase = int(math.sqrt(target_area / aspect_ratio))
            
            if h_erase < h and w_erase < w:
                x = torch.randint(0, w - w_erase + 1, (1,)).item()
                y = torch.randint(0, h - h_erase + 1, (1,)).item()
                
                # 랜덤한 gray value로 채움
                erase_value = torch.FloatTensor(1).uniform_(0, 255).item()
                img[:, y:y+h_erase, x:x+w_erase] = erase_value
        
        return img

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx):
        filename, raw_boxes = self.infos[idx]
        image_path = os.path.join(self.image_dir, filename)
        
        image_pil = self._process_image(image_path)
        if image_pil is None:
            return self.__getitem__((idx + 1) % len(self))

        # 텐서로 변환
        img = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() # CHW

        # 라벨 준비: [xmin, ymin, xmax, ymax, class_idx]
        labels = []
        for box in raw_boxes:
            x, y, w, h = box
            labels.append([x, y, x+w, y+h, 1.0]) # Class 1 for face
        
        if len(labels) == 0:
            labels = torch.zeros((0, 5), dtype=torch.float32)
        else:
            labels = torch.tensor(labels, dtype=torch.float32)

        # 테스트 split이거나 box가 없으면 증강 스킵
        if self.split == 'train' and len(labels) > 0:
            img, labels = self._crop(img, labels)
            img, labels = self._pad_to_square(img, labels)
        
        img, labels = self._resize(img, labels)

        # TEST split이거나 box가 없으면 증강 스킵
        if self.split != 'test' and len(labels) > 0:
            valid_mask = (labels[:, 2] > labels[:, 0]) & (labels[:, 3] > labels[:, 1])
            labels = labels[valid_mask]
            
            if len(labels) == 0:
                return self.__getitem__((idx + 1) % len(self))

        if self.split == 'train' and len(labels) > 0:
            # 증강 강화
            img, labels = self._flip(img, labels)
            
            # 추가 증강을 여러 개 적용 (noise와 erasing 추가)
            aug_choice = torch.rand(1).item()
            if aug_choice < 0.4:  # 40% - color distortion
                img = self._distort(img)
            elif aug_choice < 0.6:  # 20% - blur
                img = self._blur(img)
            elif aug_choice < 0.8:  # 20% - noise
                img = self._add_noise(img)
            elif aug_choice < 0.95:  # 15% - erasing
                img = self._random_erasing(img)
            # 5% - 추가 증강 없음 (flip만)

        if self.boxes is not None:
            labels = encode_pt(labels, self.boxes)

        img = img / 255.0

        return img, labels

'''
기본 앵커박스에 관한 고찰
정확히는 SSD 논문 용어로:
Prior Box
Default Box

이라 부르는 것인데
일반적인 anchor 방식과 동일하게 동작하므로 Anchor Box라고 부르겠다.

wideface 데이터셋의 경우 논문에서 말하던 small face 비율이 높다는 게 숫자로 확인해보니 
입력을 256×320으로 리사이즈한다고 가정하면,

default box에서 10, 16, 24, 32, 48 픽셀 anchor들이
이 분포의 bulk(짧은 변 6~40px)를 대부분 커버하는 구조가 된다.

이것이 왜 좋은지 생각해보면, 평균적으로는 30×37 정도지만, median이 16×21 근처

width mean: 29.00 / std: 46.25
height mean: 37.44 / std: std: 61.04

aspect ratio (w / h) / mean: 0.80(얼굴은 항상 길쭉) / std: 0.17


'''


def default_box(image_height=256, image_width=320):
    BOX_MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    BOX_STEPS = [8, 16, 32, 64]
    image_sizes = (image_height, image_width)
    min_sizes = BOX_MIN_SIZES
    steps = BOX_STEPS
    
    feature_maps = [
        [math.ceil(image_sizes[0] / step), math.ceil(image_sizes[1] / step)]
        for step in steps
    ]
    from itertools import product
    boxes = []
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes[k]:
                s_kx = min_size / image_sizes[1]
                s_ky = min_size / image_sizes[0]
                cx = (j + 0.5) * steps[k] / image_sizes[1]
                cy = (i + 0.5) * steps[k] / image_sizes[0]
                boxes += [cx, cy, s_kx, s_ky]
    
    boxes = np.asarray(boxes).reshape([-1, 4])
    return torch.tensor(boxes, dtype=torch.float32)
