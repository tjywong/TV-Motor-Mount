import os
import cv2
from ultralytics import YOLO

# Pose model detects people plus 17 body keypoints (downloaded automatically on first run)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n-pose.pt")

# COCO keypoint indices for the head: nose, left eye, right eye, left ear, right ear
HEAD_KEYPOINTS = [0, 1, 2, 3, 4]
KEYPOINT_CONF = 0.5

# Tracking modes: SINGLE dots the closest person's head, MULTI dots the average of all heads
SINGLE, MULTI = "single", "multi"

MAX_FAILED_READS = 30

# Cameras to use, by name, in order of preference (first one connected wins)
CAMERA_NAMES = ["Innomaker", "FaceTime HD Camera"]

def find_camera_index(names):
    # OpenCV numbers AVFoundation cameras in order of their unique IDs, so list the
    # cameras the same way and return the index of the first preferred name found
    try:
        import AVFoundation as AVF
    except ImportError:
        print("[WARN] pyobjc-framework-AVFoundation not installed; using camera 0.")
        return 0
    device_types = [AVF.AVCaptureDeviceTypeBuiltInWideAngleCamera]
    for t in ("AVCaptureDeviceTypeExternal", "AVCaptureDeviceTypeExternalUnknown",
              "AVCaptureDeviceTypeContinuityCamera"):
        if hasattr(AVF, t):
            device_types.append(getattr(AVF, t))
    session = AVF.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        device_types, AVF.AVMediaTypeVideo, AVF.AVCaptureDevicePositionUnspecified)
    devices = sorted(session.devices(), key=lambda d: d.uniqueID())
    for name in names:
        for index, device in enumerate(devices):
            if name.lower() in device.localizedName().lower():
                print(f"[INFO] Using camera {index}: {device.localizedName().strip()}")
                return index
    print("[WARN] None of the preferred cameras found; using camera 0.")
    return 0

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

    # Open webcam using macOS AVFoundation backend
    cap = cv2.VideoCapture(find_camera_index(CAMERA_NAMES), cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Webcam couldn't be opened. Check System Settings > Privacy & Security > Camera.")
        return

    print("[INFO] Running YOLOv8 Person Detection. Press 'm' to switch mode, 'q' to quit.")

    mode = SINGLE
    failed_reads = 0
    while True:
        ret, frame = cap.read()
        # Cameras often drop frames while warming up; only give up after many failures in a row
        if not ret:
            failed_reads += 1
            if failed_reads >= MAX_FAILED_READS:
                print("[ERROR] Camera stopped returning frames.")
                break
            continue
        failed_reads = 0

        # Run YOLOv8 inference
        results = model(frame, stream=True, verbose=False)

        person_count = 0
        heads = []  # (head center, box area) for each person whose head is visible
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

                    if kpts is not None and kpts.conf is not None:
                        head = head_center(kpts.xy[i], kpts.conf[i])
                        if head is not None:
                            heads.append((head, (x2 - x1) * (y2 - y1)))

        # Single mode: largest box is usually the closest person. Multi mode: average of all heads.
        target = None
        if heads and mode == SINGLE:
            target = max(heads, key=lambda h: h[1])[0]
        elif heads and mode == MULTI:
            target = (int(sum(h[0][0] for h in heads) / len(heads)),
                      int(sum(h[0][1] for h in heads) / len(heads)))
        if target is not None:
            cv2.circle(frame, target, 6, (0, 0, 255), -1)

        cv2.putText(frame, f'Total Persons: {person_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f'Mode: {mode} (m to switch)', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("YOLOv8 Person Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('m'):
            mode = MULTI if mode == SINGLE else SINGLE

    cap.release()
    cv2.destroyAllWindows()
    # macOS needs extra event-loop pumps for the window to actually close
    for _ in range(4):
        cv2.waitKey(1)
    print("[INFO] Detection complete.")

if __name__ == "__main__":
    detect_person_yolov8()
