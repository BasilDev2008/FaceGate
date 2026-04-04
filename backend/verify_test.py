import os
import cv2
import numpy as np
from face_engine import FaceEngine
from database import Database


def draw_modern_label(frame, box, name, confidence, recognized):
    x, y, w, h = box
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    # box color
    if recognized:
        box_color = (60, 200, 120)   # green-ish
        label_bg = (30, 30, 30)      # dark gray
    else:
        box_color = (80, 80, 220)    # red-ish in BGR
        label_bg = (25, 25, 25)

    # draw face rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

    # label text
    line1 = name
    line2 = f"{confidence:.4f}"

    font = cv2.FONT_HERSHEY_DUPLEX
    scale1 = 0.72
    scale2 = 0.55
    thickness1 = 1
    thickness2 = 1
    padding_x = 12
    padding_y = 10
    line_gap = 8

    (tw1, th1), _ = cv2.getTextSize(line1, font, scale1, thickness1)
    (tw2, th2), _ = cv2.getTextSize(line2, font, scale2, thickness2)

    label_w = max(tw1, tw2) + padding_x * 2
    label_h = th1 + th2 + padding_y * 2 + line_gap

    label_x1 = x
    label_y2 = y - 8
    label_y1 = label_y2 - label_h

    # if not enough space above face, draw below
    if label_y1 < 0:
        label_y1 = y + h + 8
        label_y2 = label_y1 + label_h

    label_x2 = label_x1 + label_w

    # keep inside frame
    if label_x2 > frame.shape[1]:
        shift = label_x2 - frame.shape[1]
        label_x1 -= shift
        label_x2 -= shift

    overlay = frame.copy()
    cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), label_bg, -1)
    frame = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)

    # top accent line
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y1 + 4), box_color, -1)

    text1_x = label_x1 + padding_x
    text1_y = label_y1 + padding_y + th1
    text2_x = label_x1 + padding_x
    text2_y = text1_y + line_gap + th2

    cv2.putText(
        frame,
        line1,
        (text1_x, text1_y),
        font,
        scale1,
        (255, 255, 255),
        thickness1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        line2,
        (text2_x, text2_y),
        font,
        scale2,
        (210, 210, 210),
        thickness2,
        cv2.LINE_AA
    )

    return frame


def load_registered_users(db):
    users = db.get_all_users()
    if not users:
        return [], None, []

    face_embeddings = []
    user_ids = []

    for user in users:
        stored_face_embedding, _ = db.get_embedding(user)
        face_embeddings.append(stored_face_embedding[0])
        user_ids.append(user.id)

    face_matrix = np.array(face_embeddings, dtype=np.float32)
    return users, face_matrix, user_ids


def find_user_by_id(users, user_id):
    for user in users:
        if user.id == user_id:
            return user
    return None


def main():
    face_engine = FaceEngine()
    db = Database()

    users, face_matrix, user_ids = load_registered_users(db)
    if face_matrix is None or len(users) == 0:
        print("No registered users found in the database.")
        return

    print(f"Loaded {len(users)} registered user(s).")

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
    cap.set(cv2.CAP_PROP_CONTRAST, 50)
    cap.set(cv2.CAP_PROP_SATURATION, 50)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Actual FPS: {cap.get(cv2.CAP_PROP_FPS)}")

    window_name = "FaceGate Live Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    FACE_THRESHOLD = 0.60

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Could not read frame from webcam.")
            break

        try:
            faces = face_engine.extract_all_embeddings(frame)
        except Exception as e:
            print(f"Face recognition crashed: {e}")
            break

        display_frame = frame.copy()

        for embedding, box in faces:
            match_id, confidence = face_engine.compare(
                embedding,
                face_matrix,
                user_ids
            )

            matched_user = find_user_by_id(users, match_id)

            if matched_user is not None and confidence >= FACE_THRESHOLD:
                name = matched_user.name
                recognized = True
            else:
                name = "Unknown"
                recognized = False

            display_frame = draw_modern_label(
                display_frame,
                box,
                name,
                confidence,
                recognized
            )

        # header
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (430, 58), (18, 18, 18), -1)
        display_frame = cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0)

        cv2.putText(
            display_frame,
            "FaceGate Live Recognition",
            (18, 24),
            cv2.FONT_HERSHEY_DUPLEX,
            0.72,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            display_frame,
            "Press Q to quit",
            (18, 48),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (210, 210, 210),
            1,
            cv2.LINE_AA
        )

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()