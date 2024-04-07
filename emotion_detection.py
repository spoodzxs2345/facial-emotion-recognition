from ultralytics import YOLO
from ultralytics.engine.results import Results
from deepface import DeepFace
import cv2
import streamlit as st

# enable the camera to capture the video feed
cap = cv2.VideoCapture(0)

st.title('Emotion Detection using YOLOv8 and DeepFace')

frame_placeholder = st.empty()

stop_btn = st.button('Stop')

# load the YOLO model
model = YOLO('C:/Users/Delsie/Desktop/projects/FER/yolov8n-face_openvino_model') # update the path to the model

while True and not stop_btn:
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if ret:
        # detect for faces in the frame
        results: Results = model.predict(frame.copy(), imgsz=320, half=True, device='cpu', max_det = 1)[0]
        detected_objects = []

        if hasattr(results, 'boxes') and hasattr(results, 'names'):
            for box in results.boxes.xyxy:
                object_id = int(box[-1])
                object_name = results.names.get(object_id)
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                detected_objects.append((object_name, (x1, y1, x2, y2)))

        # analyze the detected faces for emotions using DeepFace
        for i, (object_name, (x1, y1, x2, y2)) in enumerate(detected_objects):
            face = frame[y1:y2, x1:x2]

            analyze = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)

            emotion = analyze[0]['dominant_emotion']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # show the feed
        #cv2.imshow('feed', frame)
        frame_placeholder.image(frame, channels='RGB')
    
    key = cv2.waitKey(1)
    if key == ord('q') or stop_btn: # press 'q' to terminate the program
        st.write('Program terminated')
        break

cap.release()
cv2.destroyAllWindows()

