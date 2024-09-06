from django.shortcuts import render
import cv2
from ultralytics import YOLO 
import os
from django.conf import settings
import math
from django.http import HttpResponse
import numpy as np
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
import joblib
from ultralytics import YOLO
import datetime

def extract_frame_features(frame, model):
    # Resize frame to (224, 224) as required by VGG16
    frame_resized = cv2.resize(frame, (224, 224))
    # Preprocess frame
    frame_preprocessed = preprocess_input(frame_resized)
    # Expand dimensions to match the input shape expected by VGG16
    frame_preprocessed = np.expand_dims(frame_preprocessed, axis=0)
    # Extract features from the frame using VGG16
    features = model.predict(frame_preprocessed)
    return features

anomaly_detected=False
def index(request):
    if request.method == 'POST':
        out = None
        cap = None

        if 'start_webcam' in request.POST:
            cap = cv2.VideoCapture(0)
            frames_to_skip = 10
        elif 'upload_video' in request.POST:
            video = request.FILES['video']
            video_path = os.path.join(settings.MEDIA_ROOT, video.name)
            with open(video_path, 'wb') as destination:
                for chunk in video.chunks():
                    destination.write(chunk)
            cap = cv2.VideoCapture(video_path)
            frames_to_skip = 10
        else:
            return HttpResponse("Invalid form submission.")

        frame_width = 800  # Set the frame width to 800
        frame_height = 600  # Set the frame height to 600
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        # Calculate total frames in the input video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(os.path.join(settings.MEDIA_ROOT, 'ATM_output.avi'),
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                              (frame_width, frame_height))

        # # Define the output video file name based on current date and time
        # current_datetime = datetime.datetime.now().strftime("%d-%m-%Y + %H:%M")
        # output_video_name = f"{current_datetime}.mp4"
        # output_video_path = os.path.join(settings.MEDIA_ROOT, output_video_name)

        # out = cv2.VideoWriter(output_video_path,
        #                       cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 10,
        #                       (frame_width, frame_height))

        # Define the number of frames to skip
        frame_count = 0

        # Load the SVM model
        clf = joblib.load(os.path.join(settings.MEDIA_ROOT, "models", "svm_model0.1.pkl"))

        # Load the VGG16 model pre-trained on ImageNet
        vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

        # Define the paths to the YOLO model files
        model_folder = os.path.join(settings.MEDIA_ROOT, "models")
        model_paths = [
            os.path.join(model_folder, 'Gun.pt'),
            os.path.join(model_folder, 'Mask_Detection.pt'),
            os.path.join(model_folder, 'motorcyclehelmet.pt')
        ]
        # Initialize YOLO models
        models = [YOLO(model_path) for model_path in model_paths]
        # Define the class names for each model
        class_names_list = [
            ["Gun"],
            ["Mask"],
            ["Helmet"]
        ]
        # Define confidence thresholds for each class
        confidence_thresholds = {"Gun": 0.82, "Mask": 0.8, "Helmet": 0.8}
        # confidence_thresholds = {"guns": 0.8}

        consecutive_neg_count = 0
        anomaly_detected = False
        alerts = set()

        while True:
            success, img = cap.read()
            if not success:
                break

            frame_count += 1
            if frame_count % frames_to_skip != 0:
                continue

            frame2 = cv2.resize(img, (frame_width, frame_height))  # Resize frame

            # Inside the loop where YOLO results are processed
            for model, class_names in zip(models, class_names_list):
                results = model(frame2, stream=True)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        confidence = box.conf[0]
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        if class_name in class_names and confidence > confidence_thresholds[class_name]:
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            cv2.rectangle(frame2, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Ensure that this line draws the bounding box
                            org = [x1, y1]
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 1
                            color = (255, 0, 0)
                            thickness = 2
                            cv2.putText(frame2, f"{class_name}: {confidence:.2f}", org, font, fontScale, color, thickness)
                            alerts.add(class_name + " Detected")

            try:
                features = extract_frame_features(img, vgg_model)
            except Exception as e:
                print(f"Error predicting features: {e}")
                continue

            label = clf.predict(features.reshape(1, -1))
            prediction_text = "Negative" if label == 0 else "Positive"
            if not anomaly_detected:
                if prediction_text == "Positive":
                    cv2.putText(frame2, prediction_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame2, prediction_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2, cv2.LINE_AA)

            print(prediction_text)
            if prediction_text == "Negative":
                consecutive_neg_count += 1
            else:
                consecutive_neg_count = 0

            if consecutive_neg_count >= 4:
                anomaly_detected = True

            if anomaly_detected:
                cv2.putText(frame2, "Alert: Anomaly Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            for i, alert in enumerate(alerts):
                cv2.putText(frame2, f"Alert: {alert}", (10, 50 + (i + 1) * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Image", frame2)
            out.write(frame2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        if cap is not None:
            cap.release()

        if out is not None:
            out.release()

        cv2.destroyAllWindows()

    return render(request, 'index.html')


# def index(request):
#     if request.method == 'POST':
#         out = None
#         cap = None

#         model_paths = [
#             os.path.join(settings.MEDIA_ROOT, 'Mask_Detection.pt'),
#             os.path.join(settings.MEDIA_ROOT, 'knifeWeaponGun.pt'),
#             os.path.join(settings.MEDIA_ROOT, 'motorcyclehelmet.pt'),
#             os.path.join(settings.MEDIA_ROOT, 'Gun.pt'),
#             os.path.join(settings.MEDIA_ROOT, 'knife.pt')
#         ]

#         class_names_list = [
#             ['mask', 'no_mask'],
#             ['knife', 'weapon', 'gun'],
#             ['helmet', 'Helmet', 'No_Helmet'],
#             ['gun'],
#             ['knife']
#         ]

#         models = [YOLO(model_path) for model_path in model_paths]

#         if 'start_webcam' in request.POST:
#             cap = cv2.VideoCapture(0)
#             frames_to_skip = 20
#         elif 'upload_video' in request.POST:
#             video = request.FILES['video']
#             video_path = os.path.join(settings.MEDIA_ROOT, video.name)
#             with open(video_path, 'wb') as destination:
#                 for chunk in video.chunks():
#                     destination.write(chunk)
#             cap = cv2.VideoCapture(video_path)
#             frames_to_skip = 25
#         else:
#             return HttpResponse("Invalid form submission.")

#         frame_width = int(cap.get(3))
#         frame_height = int(cap.get(4))
#         frame_rate = cap.get(cv2.CAP_PROP_FPS)
        

#         # Calculate total frames in the input video
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         out = cv2.VideoWriter(os.path.join(settings.MEDIA_ROOT, 'ATM_output.avi'),
#                               cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
#                               (frame_width, frame_height))

#         # Define the number of frames to skip
#         frame_count = 0

#         while True:
#             success, img = cap.read()
#             if not success:
#                 break

#             # Skip frames if needed
#             if frame_count % frames_to_skip != 0:
#                 frame_count += 1
#                 continue

#             for model, class_names in zip(models, class_names_list):
#                 results = model(img, stream=True)

#                 for r in results:
#                     boxes = r.boxes
#                     for box in boxes:
#                         x1, y1, x2, y2 = map(int, box.xyxy[0])

#                         cls = int(box.cls[0])
#                         class_name = class_names[cls]

#                         confidence = box.conf[0]

#                         if class_name in ['mask', 'knife', 'weapon', 'gun', 'Helmet'] and confidence >= 0.8:
#                             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

#                             conf = math.ceil((confidence * 100)) / 100
#                             label = f'{class_name}{conf}'

#                             t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
#                             c2 = x1 + t_size[0], y1 - t_size[1] - 3

#                             cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
#                             cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1,
#                                         lineType=cv2.LINE_AA)

#             out.write(img)
#             cv2.imshow("Image", img)

#             # # Resize the frame to 640x480 before displaying
#             # resized_img = cv2.resize(img, (640, 480))
#             # cv2.imshow("Image", resized_img)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#             frame_count += 1

#             # # Break the loop if the number of written frames equals the total frames
#             # if frame_count >= total_frames:
#             #     break

#         if cap is not None:
#             cap.release()

#         if out is not None:
#             out.release()

#         cv2.destroyAllWindows()

#     return render(request, 'index.html')
