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

tracker = CentroidTracker(maxDisappeared=100, maxDistance=90)
tracker2 = CentroidTracker(maxDisappeared=240, maxDistance=100)

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

refPt = [(106, 114), (283, 356)]
pt1 = (106, 114)
pt2 = (283, 356)

# Define the second ROI
#ROI coordinates: Top-left: (338, 10), Bottom-right: (801, 581)
#ROI coordinates: Top-left: (333, 6), Bottom-right: (782, 584)
#ROI coordinates: Top-left: (323, 0), Bottom-right: (788, 580)
#ROI coordinates: Top-left: (390, 5), Bottom-right: (792, 586)
#ROI coordinates: Top-left: (204, 5), Bottom-right: (758, 579)
#pt1=(152,532), pt2=(532, 10), pt3=(920,30), pt4=(774, 584)
# =============================================================================
# refPt_02 = [(470, 52), (770, 580)]
# pt3 = (470, 52)
# pt4 = (770, 580)
# =============================================================================

refPt_02 = [(338, 10), (801, 581)]
pt3 = (338, 10)
pt4 = (801, 581)

objectId_list1 = []
objectId_list2 =[]
dtime = dict()
dwell_time = dict()
z1_text_time = ""
frame_counter = 0


fourcc_codec = cv2.VideoWriter_fourcc(*'XVID')
fps = 10.0
capture_size = (int(cap.get(3)), int(cap.get(4)))

out_ = cv2.VideoWriter("output_multiple_roi_people_count_final_v1.mp4", fourcc_codec, fps, capture_size)


while True:
    ret, img_f = cap.read()

    if ret:
        frame1 = img_f[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]  # First ROI
        frame2 = img_f[refPt_02[0][1]:refPt_02[1][1], refPt_02[0][0]:refPt_02[1][0]]  # Second ROI

        cv2.rectangle(img_f, pt1, pt2, (255, 0, 0), 2)
        cv2.rectangle(img_f, pt3, pt4, (0, 255, 0), 2)
        

        total_frames = total_frames + 1

        (H1, W1) = frame1.shape[:2]
        (H2, W2) = frame2.shape[:2]

        blob1 = cv2.dnn.blobFromImage(frame1, 0.007843, (W1, H1), 127.5)
        blob2 = cv2.dnn.blobFromImage(frame2, 0.007843, (W2, H2), 127.5)

        # Detector and tracking for the first ROI
        detector.setInput(blob1)
        person_detections1 = detector.forward()
        rects1 = []

        for i in np.arange(0, person_detections1.shape[2]):
            confidence = person_detections1[0, 0, i, 2]
            if confidence > 0.4:
                idx = int(person_detections1[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections1[0, 0, i, 3:7] * np.array([W1, H1, W1, H1])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects1.append(person_box)

        # Detector and tracking for the second ROI
        detector.setInput(blob2)
        person_detections2 = detector.forward()
        rects2 = []

        for i in np.arange(0, person_detections2.shape[2]):
            confidence = person_detections2[0, 0, i, 2]
            if confidence > 0.7:
                idx = int(person_detections2[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections2[0, 0, i, 3:7] * np.array([W2, H2, W2, H2])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects2.append(person_box)

        # Process and display results for both ROIs
        boundingboxes1 = np.array(rects1).astype(int)
        boundingboxes2 = np.array(rects2).astype(int)
        rects1 = non_max_suppression_fast(boundingboxes1, 0.3)
        rects2 = non_max_suppression_fast(boundingboxes2, 0.3)

        objects1 = tracker.update(rects1)
        objects2 = tracker2.update(rects2)

        # Draw rectangles and display results for the first ROI
        for (objectId, bbox) in objects1.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # Frame time calculation for the first ROI
            if objectId not in objectId_list1:
                objectId_list1.append(objectId)
                dtime[objectId] = datetime.datetime.now()
                dwell_time[objectId] = 0
            else:
                curr_time = datetime.datetime.now()
                old_time = dtime[objectId]
                time_diff = curr_time - old_time
                dtime[objectId] = datetime.datetime.now()
                sec = time_diff.total_seconds()
                dwell_time[objectId] += sec

            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #text = "id:{},ft:{}sec".format(objectId, int(dwell_time[objectId]))
            text = "Zone-1"
            cv2.putText(frame1, text, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            #z1_text_time = "Z1_Time:{} mins".format(int(dwell_time[objectId])// 60)
            #cv2.putText(img_f, z1_text_time, (850, 175), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, (0, 0, 255), 1)
            # Update z1_text_time inside the loop
            z1_text_time = "Z1_Time:{} mins".format(int(dwell_time[objectId])//60)
        
        # Display z1_text_time outside the loop
        cv2.putText(img_f, z1_text_time, (850, 175), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, (0, 0, 255), 2)

        # Draw rectangles and display results for the second ROI
        for (objectId2, bbox2) in objects2.items():
            x1, y1, x2, y2 = bbox2
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # Frame time calculation for the second ROI
            if objectId2 not in objectId_list2:
                objectId_list2.append(objectId2)

            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #text2 = "id:{}".format(objectId2)
            text2 = "Zone-2"
            cv2.putText(frame2, text2, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        
        #count current perosn count and total person count---zone 1
        opc_count1 = len(objectId_list1)
        opc_txt1 = "Zone-1: {}".format(opc_count1)
        cv2.putText(img_f, opc_txt1, (850, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 0), 2)

        #count current perosn count and total person count---zone 2     
        opc_count2 = len(objectId_list2)

        if opc_count2 <= 2:
            opc_txt2 = "Zone-2: {}".format(opc_count2)  # Display 2 if opc_count2 is 2 or less
        elif opc_count2== 2:
            opc_txt2 = "Zone-2: {}".format(2)
        
        cv2.putText(img_f, opc_txt2, (850, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (50, 255, 50), 2)

        
        #z1_text_time = "Z1_Time:{} mins".format(int(dwell_time[objectId]) // 60)
        #cv2.putText(img_f, z1_text_time, (850, 175), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, (0, 0, 255), 1)
        
        #write frame
        out_.write(img_f)

        cv2.imshow("Application", img_f)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()




