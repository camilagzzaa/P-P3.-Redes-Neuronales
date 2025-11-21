import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import argparse
import os

# --------- Parámetros ----------
MODEL_PATH = "best_model_numbers_v2.h5"  # Ya coincide con tu notebook
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_INPUT_SIZE = (28, 28)
RECORD_SECONDS = 30
OUTPUT_VIDEO = "reconocimiento_video.mp4"
MIN_CONTOUR_AREA = 200
DILATION_ITER = 1
# -------------------------------

def preprocess_roi(roi):
    if len(roi.shape) == 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi.copy()
    roi_resized = cv2.resize(roi_gray, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    roi_norm = roi_resized.astype("float32") / 255.0
    roi_expanded = np.expand_dims(roi_norm, axis=(0, -1))
    return roi_expanded, roi_resized

def overlay_probabilities(frame, probs, classes, pos=(10,30)):
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    line_h = 18
    winner = int(np.argmax(probs))
    for i, cls in enumerate(classes):
        prob = probs[i]
        text = f"{cls}: {prob:.2f}"
        color = (0,255,0) if i == winner else (0,0,255)
        cv2.putText(frame, text, (x, y + i*line_h), font, scale, color, 1, cv2.LINE_AA)

def find_and_predict(frame, model, classes):
    detections = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(th, kernel, iterations=DILATION_ITER)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        roi = frame[y:y+h, x:x+w]
        roi_input, roi_resized = preprocess_roi(roi)
        preds = model.predict(roi_input)
        probs = preds[0]
        label_idx = int(np.argmax(probs))
        label = classes[label_idx]
        detections.append((x,y,w,h,label,probs[label_idx], probs, roi_resized))
    return detections, dilated

def main():
    print("Cargando modelo:", MODEL_PATH)
    model = load_model(MODEL_PATH, compile=False)

    classes = [str(i) for i in range(10)]

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

    start_time = time.time()
    print("Grabando... presiona 'q' para terminar antes de 30s.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_disp = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        detections, mask = find_and_predict(frame_disp, model, classes)

        global_probs = np.zeros((10,), float)

        if len(detections) > 0:
            probs_stack = np.stack([d[6] for d in detections])
            global_probs = np.mean(probs_stack, axis=0)

            for (x,y,w,h,label,val,p_array,roi_resized) in detections:
                cv2.rectangle(frame_disp, (x,y), (x+w, y+h), (255,0,0), 2)
                cv2.putText(frame_disp, f"{label} ({val:.2f})", (x,y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

                roi_show = cv2.resize(roi_resized, (120,120), interpolation=cv2.INTER_NEAREST)
                frame_disp[10:130, FRAME_WIDTH-140:FRAME_WIDTH-20] = cv2.cvtColor(roi_show, cv2.COLOR_GRAY2BGR)

        overlay_probabilities(frame_disp, global_probs, classes)

        cv2.imshow("Clasificación en tiempo real", frame_disp)
        out.write(frame_disp)

        if time.time() - start_time > RECORD_SECONDS:
            print("Tiempo de grabación completado.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Detenido por el usuario.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video guardado como:", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
