import cv2
import dlib

cap = cv2.VideoCapture(0)
detector_hog = dlib.get_frontal_face_detector()
model_path = 'models/shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dlib_rects = detector_hog(img_rgb, 1)

    list_landmarks = []  # ← 매 프레임마다 초기화

    for dlib_rect in dlib_rects:
        cv2.rectangle(img_rgb, (dlib_rect.left(), dlib_rect.top()),
                      (dlib_rect.right(), dlib_rect.bottom()), (0, 255, 0), 2, lineType=cv2.LINE_AA)

        points = landmark_predictor(img_rgb, dlib_rect)
        list_points = [(p.x, p.y) for p in points.parts()]
        list_landmarks.append(list_points)

    img_show_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    for landmark in list_landmarks:
        for point in landmark:
            cv2.circle(img_show_rgb, point, 2, (0, 255, 255), -1)

    cv2.imshow('Webcam Stream', img_show_rgb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
