# Semantic segmentation mask

### 1. **CRF(Conditional Random Field) 후처리**

모델이 만든 마스크는 주변 픽셀 관계를 잘 반영하지 못해서 경계가 흐릿하게 나올 때가 많습니다.

CRF는 인접한 픽셀끼리 비슷한 색이나 위치에 있으면 같은 클래스로 묶고,

색이 급격히 변하는 부분은 다른 클래스로 나누어 줍니다.

그래서 결과적으로 경계가 훨씬 자연스럽고 선명해집니다.

DeepLab 시리즈에서도 실제로 이 방식을 사용했습니다.

---

### 2. **Boundary loss나 Edge-aware loss 사용**

일반적인 Cross Entropy는 전체 픽셀 평균만 보기 때문에

경계 부분의 작은 오류를 잘 잡지 못합니다.

Boundary loss는 경계 근처 픽셀의 손실을 더 크게 반영해서

모델이 경계 정보를 더 신경 쓰게 만듭니다.

결과적으로 객체와 배경의 구분선이 또렷해집니다.

---

### 3. **Skip connection 구조 사용**

Pooling으로 특징을 추출하면 세부 정보가 사라집니다.

그래서 U-Net처럼 초기 레이어의 세밀한 특징을

뒤쪽 레이어의 의미적 특징과 합쳐주는 구조를 사용합니다.

이렇게 하면 고해상도 세부 정보와 문맥 정보를 동시에 반영할 수 있습니다.

---

### 4. **데이터 품질과 증강**

라벨 자체가 부정확하면 모델도 잘못된 정보를 배우게 됩니다.

라벨을 정교하게 다듬거나, 밝기·대비·노이즈 같은 증강을 추가하면

모델이 다양한 경계 상황에 더 잘 적응할 수 있습니다.

---

### 5. **다중 스케일 특징 활용 (ASPP, FPN 등)**

객체 크기가 다양하거나 배경이 복잡하면 하나의 스케일만으로는 한계가 있습니다.

여러 크기의 필터(dilation rate)를 병렬로 적용하면

넓은 문맥과 세부 정보를 동시에 볼 수 있습니다.

그래서 작은 객체도 놓치지 않고 큰 배경도 덜 헷갈립니다.

---

**정리하자면**,

시맨틱 마스크 오류를 줄이려면 **경계를 정교하게 다루는 방법(CRF, Boundary loss)** 과

**해상도를 보존하는 구조(U-Net, ASPP)** 가 핵심입니다.

이 중에서도 CRF는 가장 간단하면서 효과적인 후처리 방법으로 자주 쓰입니다.

---

### 가장 현실적인 솔루션 제안

https://pseudo-lab.github.io/SegCrew-Book/docs/Appendix/DenseCRF.html

DenseCRF 후처리

```python
# pip install pydensecrf
import numpy as np
import torch
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def crf_refine(rgb_img: np.ndarray, prob: np.ndarray,  # prob: (C,H,W), softmax 확률
               sxy_gauss=3, compat_gauss=3,
               sxy_bi=80, srgb_bi=13, compat_bi=10):

    H, W = rgb_img.shape[:2]
    C = prob.shape[0]

    d = dcrf.DenseCRF2D(W, H, C)
    unary = unary_from_softmax(prob)  # shape: (C, H*W), negative log
    d.setUnaryEnergy(unary)

    
    d.addPairwiseGaussian(sxy=sxy_gauss, compat=compat_gauss)

   
    d.addPairwiseBilateral(sxy=sxy_bi, srgb=srgb_bi,
                           rgbim=rgb_img, compat=compat_bi)

    Q = d.inference(5)  # iterations
    return np.array(Q).reshape(C, H, W).argmax(0).astype(np.int32)

# 예시: PyTorch 모델 출력(logits)을 CRF로 다듬기
def refine_batch_with_crf(images_uint8, logits):
    """
    images_uint8: (B,H,W,3) uint8
    logits:       (B,C,H,W) float32
    return:       (B,H,W) int32
    """
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    imgs = images_uint8
    outs = []
    for i in range(probs.shape[0]):
        outs.append(crf_refine(imgs[i], probs[i]))  
    return np.stack(outs, 0)
```

- 얇은 경계가 누수되면 `compat_bi`↑ 또는 `srgb_bi`↓.
- 과도한 스무딩이면 `compat_gauss`↓ 또는 `sxy_gauss`↓.
- 속도가 느리면 `inference` 반복 수(5→3)↓.