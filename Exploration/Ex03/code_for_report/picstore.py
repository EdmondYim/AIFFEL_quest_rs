import cv2
import os

save_dir = 'captured_frames'
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

frame_count = 0

print("화면에서 'c'를 누르면 캡처, 'q'를 누르면 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1) & 0xFF

    # 'c' 키 누르면 저장
    if key == ord('c'):
        filename = os.path.join(save_dir, f'frame_{frame_count:05d}.jpg')
        cv2.imwrite(filename, frame)
        print(f"{filename} 저장 완료")
        frame_count += 1

    # 'q' 키 누르면 종료
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
