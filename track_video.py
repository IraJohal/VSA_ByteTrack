from algorithm.object_detector import YOLOv7
from utils.detections import draw
from tqdm import tqdm
import cv2
from byte_tracker import BYTETracker

yolov7 = YOLOv7()
yolov7.load('best_felix.pt', classes='coco.yaml', device='gpu') # use 'gpu' for CUDA GPU inference

video = cv2.VideoCapture('Rover_test_30.mp4')
width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

if video.isOpened() == False:
	print('[!] error opening the video')

print('[+] tracking video...\n')
pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)

tracker_team1 = BYTETracker(track_thresh = 0.5, track_buffer = 50, match_thresh = 0.8)
tracker_team2 = BYTETracker(track_thresh = 0.5, track_buffer = 50, match_thresh = 0.8)

try:
    while video.isOpened():
        ret, frame = video.read()
        if ret == True:
            detections = yolov7.detect(frame)
	    tm1 =[]
            tm2 =[]
	    for det in detection:
            	if yolov7.classes[int(det[5])].get('name') == 'team1':
              		tm1.append(det)
            	else:
              		tm2.append(det)
            tm1= np.array(tm1)
            tm2 = np.array(tm2)
            detections_team1 = tracker_team1.update(tm1)
            detections_team2 = tracker_team2.update(tm2)
            detections = yolov7.detect_2(detections_team1)
            detections_2 = yolov7.detect_2(detections_team2)
            detected_frame = draw(frame, detections)
            detected_frame = draw(detected_frame, detections_2)
	    detections = tracker.update(detections)
	    detections = yolov7.detect_2(detections)
	    detected_frame = draw(frame, detections)
	    output.write(detected_frame)
            pbar.update(1)
        else:
            break
except KeyboardInterrupt:
    pass

pbar.close()
video.release()
output.release()
yolov7.unload()
