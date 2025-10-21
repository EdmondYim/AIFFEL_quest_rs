import cv2
import dlib
import glob
import os
import csv

# 경로 설정
img_dir = 'captured_frames/extra_low'
model_path = r'models/shape_predictor_68_face_landmarks.dat'
csv_path = 'detection_result_extra_low.csv'

# 모델 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

# 이미지 목록
img_paths = glob.glob(os.path.join(img_dir, '*.*'))

# CSV 헤더 작성
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'faces', 'landmarks_detected'])

    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[X] 파일 읽기 실패: {path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rects = detector(img_rgb, 1)
        face_count = len(rects)
        landmarks_detected = False

        if face_count > 0:
            for rect in rects:
                try:
                    predictor(img_rgb, rect)
                    landmarks_detected = True
                    break  # 하나만 감지되어도 True로 처리
                except:
                    landmarks_detected = False

        filename = os.path.basename(path)
        writer.writerow([filename, face_count, landmarks_detected])
        print(f"{filename} -> 얼굴: {face_count}개, 랜드마크 감지: {landmarks_detected}")

print(f"\n결과 저장 완료: {csv_path}")
