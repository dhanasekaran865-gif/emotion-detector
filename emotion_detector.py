import cv2
from deepface import DeepFace


camera = cv2.VideoCapture(0)


while True:

    success, frame = camera.read()

    if not success:
        break


    result = DeepFace.analyze(
    img_path=frame,
    actions=["emotion"],
    detector_backend="opencv",
    enforce_detection=False,
    silent=True
)


    emotion = result[0]['dominant_emotion']


    cv2.putText(
        frame,
        emotion,
        (50,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0,255,0),
        3
    )


    cv2.imshow(
        "Live Emotion Detector",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



camera.release()

cv2.destroyAllWindows()