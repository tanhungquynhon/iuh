import time
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import argparse
import numpy as np
from imutils.video import VideoStream
import imutils

# Cai dat tham so doc weight, config va class name
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', default='yolov3-tiny.cfg',
                help='path to yolo config file')
ap.add_argument('-w', '--weights', default='yolov3-tiny_86000.weights',
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', default='yolo.names',
                help='path to text file containing class names')
args = ap.parse_args()

# Ham tra ve output layer
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Ham ve cac hinh chu nhat va ten class
def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    #cv2.putText(image, "Numbers: " + str()  , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Doc ten cac class
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(args.weights, args.config)
# Doc tu webcam
#cap  = VideoStream(src="tom-boi.mp4").start()
cap  = VideoStream(src=0).start()
start_time = time.time()
frames = 0
# Bat dau doc tu webcam
i=1
while (True):

    # Doc frame
    frame = cap.read()
    frames += 1
    image = imutils.resize(frame, width=640)
    i+=1
    if i%10==0:
        # Resize va dua khung hinh vao mang predict
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(image, scale, (280, 180), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        # Loc cac object trong khung hinh
        class_ids = []
        confidences = []
        boxes = []
        final_box = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if (confidence > 0.5):
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        info = [('So luong tom: ','{}'.format(len(indices)))]
        for (i,(txt,val)) in enumerate(info): #trich xuat phan tu trong file val (value)
            text = '{}: {}'.format(txt,val) 
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) #print màn hình

        # Ve cac khung chu nhat quanh doi tuong
        for i in indices:
            box = boxes[i]
            confi = confidences[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            final_box.append(box)
            draw_prediction(image, class_ids[i], round(x), round(y), round(x + w), round(y + h))
        end_time = time.time() - start_time
        fps =  frames/ end_time
        cv2.putText(image,"FPS: " + str(round(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("Object detection", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()


