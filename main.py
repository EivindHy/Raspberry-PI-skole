import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_match = False

reference_img = cv2.imread("image.jpg")

def check_face(frame):
    try:
        result = DeepFace.verify(frame, reference_img.copy())
        return result["verified"]
    except Exception as e:
        print("Error during face verification:", e)
        return False

while True: 
    ret, frame = cap.read()

    if ret:
        face_match = check_face(frame)

        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No match", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
