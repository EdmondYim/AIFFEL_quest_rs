import io
import os
from PIL import Image
import torch

def generate_ptexample(anno):
    filename = anno['filename']
    filepath = anno['filepath']

    # 이미지 파일 읽기
    with open(filepath, 'rb') as image_file:
        content = image_file.read()

    image = Image.open(filepath)
    # JPEG 형식 및 RGB 모드가 아니면 변환
    if image.format != 'JPEG' or image.mode != 'RGB':
        image_rgb = image.convert('RGB')
        with io.BytesIO() as output:
            image_rgb.save(output, format="JPEG", quality=95)
            content = output.getvalue()

    width, height = image.size
    depth = 3

    c_x = int(anno['center'][0])
    c_y = int(anno['center'][1])
    scale = anno['scale']

    x = [int(joint[0]) if joint[0] >= 0 else int(joint[0])
         for joint in anno['joints']]
    y = [int(joint[1]) if joint[1] >= 0 else int(joint[0])
         for joint in anno['joints']]

    v = [0 if joint_v == 0 else 2 for joint_v in anno['joints_visibility']]

    feature = {
        'image/height': height,
        'image/width': width,
        'image/depth': depth,
        'image/object/parts/x': x,
        'image/object/parts/y': y,
        'image/object/center/x': c_x,
        'image/object/center/y': c_y,
        'image/object/scale': scale,
        'image/object/parts/v': v,
        'image/encoded': content,
        'image/filename': filename.encode()  # bytes로 저장
    }

    return feature

def chunkify(l, n):
    size = len(l) // n
    start = 0
    results = []
    for i in range(n):
        results.append(l[start:start + size])
        start += size
    return results

def calculate_crop_box(keypoint_x, keypoint_y, scale, img_width, img_height, margin=0.2):
    """
    Keypoints와 Scale을 기반으로 Crop할 박스 좌표를 계산합니다.
    keypoint_x, keypoint_y: 1D tensor or list
    scale: float or tensor
    """
    # Tensor 변환 (list 입력 대응)
    if not torch.is_tensor(keypoint_x):
        keypoint_x = torch.tensor(keypoint_x, dtype=torch.float32)
    if not torch.is_tensor(keypoint_y):
        keypoint_y = torch.tensor(keypoint_y, dtype=torch.float32)
    
    if torch.is_tensor(scale):
        scale = scale.item()

    body_height = scale * 200.0

    # 유효한 keypoint (값이 0보다 큰 값)만 선택
    masked_keypoint_x = keypoint_x[keypoint_x > 0]
    masked_keypoint_y = keypoint_y[keypoint_y > 0]

    if len(masked_keypoint_x) == 0:
        # 유효한 키포인트가 없는 경우 전체 이미지 반환
        return 0, 0, img_width, img_height, 0, 0, img_width, img_height

    # 최소, 최대 값 계산
    keypoint_xmin = masked_keypoint_x.min()
    keypoint_xmax = masked_keypoint_x.max()
    keypoint_ymin = masked_keypoint_y.min()
    keypoint_ymax = masked_keypoint_y.max()

    # margin을 적용하여 경계를 확장
    extra = int(body_height * margin)
    xmin = int(keypoint_xmin.item()) - extra
    xmax = int(keypoint_xmax.item()) + extra
    ymin = int(keypoint_ymin.item()) - extra
    ymax = int(keypoint_ymax.item()) + extra

    # 이미지 경계를 벗어나지 않도록 조정
    effective_xmin = max(xmin, 0)
    effective_ymin = max(ymin, 0)
    effective_xmax = min(xmax, img_width)
    effective_ymax = min(ymax, img_height)

    return effective_xmin, effective_ymin, effective_xmax, effective_ymax, xmin, ymin, xmax, ymax


def crop_roi(image, features, margin=0.2):
    # Check if image is PIL Image
    is_pil = isinstance(image, Image.Image)
    
    if is_pil:
        img_width, img_height = image.size
    else:
        img_height, img_width, img_depth = image.shape

    keypoint_x = torch.tensor(features['image/object/parts/x'], dtype=torch.int32)
    keypoint_y = torch.tensor(features['image/object/parts/y'], dtype=torch.int32)
    scale = features['image/object/scale']

    effective_xmin, effective_ymin, effective_xmax, effective_ymax, _, _, _, _ = calculate_crop_box(
        keypoint_x, keypoint_y, scale, img_width, img_height, margin
    )

    if is_pil:
        cropped_image = image.crop((effective_xmin, effective_ymin, effective_xmax, effective_ymax))
        new_width, new_height = cropped_image.size
    else:
        cropped_image = image[effective_ymin:effective_ymax, effective_xmin:effective_xmax, :]
        new_height, new_width, _ = cropped_image.shape

    # keypoint 좌표를 정규화 (0~1 범위)
    # 0으로 나누는 것을 방지
    if new_width == 0: new_width = 1
    if new_height == 0: new_height = 1
    
    effective_keypoint_x = (keypoint_x.float() - effective_xmin) / new_width
    effective_keypoint_y = (keypoint_y.float() - effective_ymin) / new_height

    return cropped_image, effective_keypoint_x, effective_keypoint_y

def generate_2d_gaussian(height, width, y0, x0, visibility=2, sigma=1, scale=12):
    # (height, width) 크기의 0으로 채워진 heatmap 생성
    heatmap = torch.zeros((height, width), dtype=torch.float32)

    xmin = x0 - 3 * sigma
    ymin = y0 - 3 * sigma
    xmax = x0 + 3 * sigma
    ymax = y0 + 3 * sigma

    # 범위가 이미지 내에 없거나, visibility가 0이면 heatmap 그대로 반환
    if xmin >= width or ymin >= height or xmax < 0 or ymax < 0 or visibility == 0:
        return heatmap

    size = int(6 * sigma + 1)
    grid_range = torch.arange(0, size, dtype=torch.float32)
    x_grid, y_grid = torch.meshgrid(grid_range, grid_range, indexing='xy')
    center_x = size // 2
    center_y = size // 2

    # 가우시안 patch 계산
    gaussian_patch = torch.exp(-(((x_grid - center_x)**2 + (y_grid - center_y)**2) / (sigma**2 * 2))) * scale

    # 이미지와 patch 간의 겹치는 영역 계산
    patch_xmin = max(0, -xmin)
    patch_ymin = max(0, -ymin)
    patch_xmax = min(xmax, width) - xmin
    patch_ymax = min(ymax, height) - ymin

    heatmap_xmin = max(0, xmin)
    heatmap_ymin = max(0, ymin)
    heatmap_xmax = min(xmax, width)
    heatmap_ymax = min(ymax, height)

    # 계산된 영역에 gaussian_patch 값을 할당
    heatmap[heatmap_ymin:heatmap_ymax, heatmap_xmin:heatmap_xmax] = \
        gaussian_patch[int(patch_ymin):int(patch_ymax), int(patch_xmin):int(patch_xmax)]

    return heatmap

def make_heatmaps(features, keypoint_x, keypoint_y, heatmap_shape):
    # heatmap_shape: (height, width, num_heatmap)
    v = torch.tensor(features['image/object/parts/v'], dtype=torch.float32)
    # x 좌표는 width(heatmap_shape[1])에 비례
    x = torch.round(keypoint_x.float() * heatmap_shape[1]).to(torch.int32)
    # y 좌표는 height(heatmap_shape[0])에 비례
    y = torch.round(keypoint_y.float() * heatmap_shape[0]).to(torch.int32)

    num_heatmap = heatmap_shape[2]
    heatmaps_list = []
    for i in range(num_heatmap):
        # generate_2d_gaussian(height, width, ...)
        gaussian = generate_2d_gaussian(
            heatmap_shape[0], # height
            heatmap_shape[1], # width
            int(y[i].item()),
            int(x[i].item()),
            visibility=int(v[i].item())
        )
        heatmaps_list.append(gaussian)

    # (num_heatmap, height, width) 텐서 생성
    heatmaps = torch.stack(heatmaps_list, dim=0)
    
    return heatmaps
