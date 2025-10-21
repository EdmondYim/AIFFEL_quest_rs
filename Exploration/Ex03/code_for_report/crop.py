import cv2
import os
import glob
import numpy as np

src_dir = 'captured_frames'
dst_dir = 'augmented_fixed'
os.makedirs(dst_dir, exist_ok=True)


def adjust_brightness(img, alpha=0.7):
    """밝기 조절 (alpha>1 밝게, <1 어둡게)"""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def affine_transform(img, rot_deg=10, shear_deg=8):
    """회전(rot_deg도), 기울이기(shear_deg도)"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # 회전 행렬
    M_rot = cv2.getRotationMatrix2D(center, rot_deg, 1.0)

    # shear 행렬
    shear = np.tan(np.deg2rad(shear_deg))
    M_shear = np.array([[1, shear, -shear * center[1]],
                        [0, 1, 0]], dtype=np.float32)

    rotated = cv2.warpAffine(
        img, M_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    skewed = cv2.warpAffine(rotated, M_shear, (w, h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return skewed


for path in glob.glob(os.path.join(src_dir, '*.jpg')):
    img = cv2.imread(path)
    if img is None:
        continue

    # 밝기 + 회전/기울이기 고정값 적용
    img_aug = adjust_brightness(img, alpha=0.1)  # 밝기 1.3배
    img_aug = affine_transform(img_aug, rot_deg=-10, shear_deg=8)

    name = os.path.basename(path)
    save_path = os.path.join(dst_dir, f"aug_{name}")
    cv2.imwrite(save_path, img_aug)
    print(f"{save_path} 저장 완료")

print("고정값 증강 완료")
