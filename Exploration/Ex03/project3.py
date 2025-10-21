import cv2
import dlib
import numpy as np


'''
실시간으로 객체를 얼굴에 가져오지 못하면
카메라 앱인지?
'''


sticker_path = r'images/cat-whiskers.png'  # PNG(알파) 권장
model_path = r'models/shape_predictor_68_face_landmarks.dat'

# 고정 리소스는 한 번만 로드
stk0 = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)  # BGRA면 채널=4


det = dlib.get_frontal_face_detector()
pred = dlib.shape_predictor(model_path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("웹캠 시작. 'q'를 눌러 종료하세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_show = frame.copy()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴 검출
    rects = det(img_rgb, 1)

    list_landmarks = []  # 매 프레임마다 초기화

    for rect in rects:
        # 바운딩 박스 그리기 (RGB 이미지에)
        cv2.rectangle(img_show,
                      (rect.left(), rect.top()),
                      (rect.right(), rect.bottom()),
                      (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # 랜드마크 검출
        shape = pred(img_rgb, rect)
        pts = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.int32)
        list_landmarks.append(pts)

        # 랜드마크 33번 (코 중앙 하단)
        nose_tip = pts[33]
        cx, cy = nose_tip

        # 얼굴 크기 기반 스티커 크기 계산
        face_width = rect.width()
        face_height = rect.height()

        # 스티커 크기를 얼굴 너비의 80%로 설정 (자연스러운 비율)
        sticker_width = int(face_width * 0.8)
        sticker_height = int(sticker_width * (stk0.shape[0] / stk0.shape[1]))

        # 스티커 위치 계산 (랜드마크 33번을 중심으로)
        left = cx - sticker_width // 2
        top = cy - sticker_height // 2

        # 스티커 리사이즈
        stk = cv2.resize(stk0, (sticker_width, sticker_height),
                         interpolation=cv2.INTER_AREA)

        # 화면 경계 클리핑
        H, W = img_show.shape[:2]

        # 스티커가 화면 안에 들어오는 영역 계산
        y0 = max(0, top)
        x0 = max(0, left)
        y1 = min(H, top + sticker_height)
        x1 = min(W, left + sticker_width)

        # 유효한 영역인지 확인
        if y0 >= y1 or x0 >= x1:
            continue

        # 스티커의 해당 영역 크롭
        sy0 = y0 - top
        sx0 = x0 - left
        sy1 = sy0 + (y1 - y0)
        sx1 = sx0 + (x1 - x0)

        stk_crop = stk[sy0:sy1, sx0:sx1]
        roi = img_show[y0:y1, x0:x1]

        # 크기 불일치 방지
        if stk_crop.shape[:2] != roi.shape[:2]:
            continue

        # 알파 블렌딩 합성
        if stk_crop.shape[2] == 4:  # BGRA 채널
            # 알파 채널 추출 및 정규화
            alpha = stk_crop[:, :, 3:4].astype(np.float32) / 255.0

            # BGR 채널만 추출
            fg = stk_crop[:, :, :3].astype(np.float32)
            bg = roi.astype(np.float32)

            # 알파 블렌딩
            out = (alpha * fg + (1.0 - alpha) * bg).astype(np.uint8)
        else:  # BGR만 있는 경우 (대체 방법)
            gray = cv2.cvtColor(stk_crop, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
            mask_3ch = cv2.merge([mask, mask, mask])
            out = np.where(mask_3ch > 0, stk_crop, roi)

        # 결과 적용
        img_show[y0:y1, x0:x1] = out

    # 랜드마크 포인트 그리기 (모든 얼굴에 대해)
    for landmark in list_landmarks:
        for idx, point in enumerate(landmark):
            # 일반 랜드마크는 노란색으로 표시
            cv2.circle(img_show, tuple(point), 2, (0, 255, 255), -1)

    cv2.imshow('카메라 어플', img_show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
