import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker

protopath = "model/MobileNetSSD_deploy.prototxt"
modelpath = "model/MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker_zone1 = CentroidTracker(maxDisappeared=100, maxDistance=90)
tracker_zone2 = CentroidTracker(maxDisappeared=250, maxDistance=90)

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

path = "C:/Users/manish.kumar/Desktop/person_tracker_id_ft_age_gender/CCTV Recording.MP4"
#path = "C:/Users/manish.kumar/Desktop/person_tracker_id_ft_age_gender/CCTV_Recording_edited.mp4"

cap = cv2.VideoCapture(path)

fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0

objectId_list_zone1 = []
objectId_list_zone2 = []
dtime_zone1 = dict()
dtime_zone2 = dict()
dwell_time_zone1 = dict()
dwell_time_zone2 = dict()

frame_counter = 0

# Create line coordinates for both zones
line_coordinates = [(180, 570), (510, 5)]
#line_coordinates = [(4, 580), (455, 5)]

while True:
    ret, img_f = cap.read()

    if ret:
        total_frames = total_frames + 1

        (H, W) = img_f.shape[:2]

        blob = cv2.dnn.blobFromImage(img_f, 0.007843, (W, H), 127.5)

        # Detector and tracking for both zones
        detector.setInput(blob)
        person_detections = detector.forward()
        rects_zone1 = []
        rects_zone2 = []

        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.4:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")

                # Check which side of the line the person is on
                if startX + endX < line_coordinates[0][0] + line_coordinates[1][0]:
                    rects_zone1.append(person_box)
                else:
                    rects_zone2.append(person_box)

        # Process and display results for Zone 1
        boundingboxes_zone1 = np.array(rects_zone1).astype(int)
        rects_zone1 = non_max_suppression_fast(boundingboxes_zone1, 0.3)

        objects_zone1 = tracker_zone1.update(rects_zone1)

        # Draw rectangles and display results for Zone 1
        for (objectId, bbox) in objects_zone1.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # Frame time calculation for Zone 1
            if objectId not in objectId_list_zone1:
                objectId_list_zone1.append(objectId)
                dtime_zone1[objectId] = datetime.datetime.now()
                dwell_time_zone1[objectId] = 0
            else:
                curr_time = datetime.datetime.now()
                old_time = dtime_zone1[objectId]
                time_diff = curr_time - old_time
                dtime_zone1[objectId] = datetime.datetime.now()
                sec = time_diff.total_seconds()
                dwell_time_zone1[objectId] += sec

            cv2.rectangle(img_f, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "Zone 1 - id:{}, ft:{}sec".format(objectId, int(dwell_time_zone1[objectId]))
            cv2.putText(img_f, text, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        # Draw line for both zones
        cv2.line(img_f, line_coordinates[0], line_coordinates[1], (0, 255, 0), 2)

        # Count current person count and total person count for Zone 1
        opc_count_zone1 = len(objectId_list_zone1)
        opc_txt_zone1 = "Zone 1: {}".format(opc_count_zone1)
        cv2.putText(img_f, opc_txt_zone1, (850, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        # Process and display results for Zone 2
        boundingboxes_zone2 = np.array(rects_zone2).astype(int)
        rects_zone2 = non_max_suppression_fast(boundingboxes_zone2, 0.3)

        objects_zone2 = tracker_zone2.update(rects_zone2)

        # Draw rectangles and display results for Zone 2
        for (objectId, bbox) in objects_zone2.items():
            x1_2, y1_2, x2_2, y2_2 = bbox
            x1_2 = int(x1_2)
            y1_2 = int(y1_2)
            x2_2 = int(x2_2)
            y2_2 = int(y2_2)

            # Frame time calculation for Zone 2
            if objectId not in objectId_list_zone2:
                objectId_list_zone2.append(objectId)
                dtime_zone2[objectId] = datetime.datetime.now()
                dwell_time_zone2[objectId] = 0
            else:
                curr_time = datetime.datetime.now()
                old_time = dtime_zone2[objectId]
                time_diff = curr_time - old_time
                dtime_zone2[objectId] = datetime.datetime.now()
                sec = time_diff.total_seconds()
                dwell_time_zone2[objectId] += sec

            cv2.rectangle(img_f, (x1_2, y1_2), (x2_2, y2_2), (255, 0, 0), 2)
            text = "Zone 2 - id:{}, ft:{}sec".format(objectId, int(dwell_time_zone2[objectId]))
            cv2.putText(img_f, text, (x1_2, y1_2 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)

        # Count current person count and total person count for Zone 2
        opc_count_zone2 = len(objectId_list_zone2)
        opc_txt_zone2 = "Zone 2: {}".format(opc_count_zone2)
        cv2.putText(img_f, opc_txt_zone2, (850, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)

        # Count current person count and total person count considering both zones
        opc_count_total = opc_count_zone1 + opc_count_zone2
        opc_txt_total = "Total: {}".format(opc_count_total)
        cv2.putText(img_f, opc_txt_total, (850, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

        # Show results
        cv2.imshow("Application", img_f)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
