import win32event
import win32api
import sys
from winerror import ERROR_ALREADY_EXISTS
mutex = win32event.CreateMutex (None, False, 'name')
last_error = win32api.GetLastError()
if last_error == ERROR_ALREADY_EXISTS:
 sys.exit(0)


import os
import cv2
from ultralytics import YOLO

# Pose model detects people plus 17 body keypoints (downloaded automatically on first run)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n-pose.pt")

# COCO keypoint indices for the head: nose, left eye, right eye, left ear, right ear
HEAD_KEYPOINTS = [0, 1, 2, 3, 4]
KEYPOINT_CONF = 0.5

def head_center(kpts_xy, kpts_conf):
    # Average the visible head keypoints; return None if the head isn't visible
    visible = [kpts_xy[i] for i in HEAD_KEYPOINTS if kpts_conf[i] > KEYPOINT_CONF]
    if not visible:
        return None
    cx = sum(float(p[0]) for p in visible) / len(visible)
    cy = sum(float(p[1]) for p in visible) / len(visible)
    return int(cx), int(cy)

def detect_person_yolov8():
    # Load pretrained YOLOv8n pose model
    model = YOLO(MODEL_PATH)  # or 'yolov8s-pose.pt' for more accuracy

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam couldn't be opened.")
        return

    print("[INFO] Running YOLOv8 Person Detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference
        results = model(frame, stream=True)

        person_count = 0
        for r in results:
            boxes = r.boxes
            kpts = r.keypoints
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                # Only keep class 'person' (COCO class id 0)
                if cls == 0 and conf > 0.5:
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person {person_count}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Draw a dot at the center of the head
                    if kpts is not None and kpts.conf is not None:
                        head = head_center(kpts.xy[i], kpts.conf[i])
                        if head is not None:
                            cv2.circle(frame, head, 6, (0, 0, 255), -1)

        cv2.putText(frame, f'Total Persons: {person_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("YOLOv8 Person Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection complete.")

if __name__ == "__main__":
    detect_person_yolov8()