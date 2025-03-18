import cv2
import numpy as np

cap = cv2.VideoCapture("My Movie.mp4")

if not cap.isOpened():
    print("Cannot open video file.")
    exit()

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=200, qualityLevel=0.2, minDistance=5)
lk_params = dict(winSize=(15, 15), maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


mask = np.zeros_like(old_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)
        frame_with_trails = cv2.add(frame, mask)
        cv2.imshow("Optical Flow Tracking", frame_with_trails)

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None

    if p0 is None or len(p0) < 10:
        p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=200, qualityLevel=0.2, minDistance=5)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
