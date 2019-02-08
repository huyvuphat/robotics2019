import cv2
import os
# note capture the video from file same name as its folder
print(cv2.__version__)
path_arr = os.getcwd().split("/")
prefix = path_arr[len(path_arr)-1]
video_input = '%s.mp4' % (prefix)
vidcap = cv2.VideoCapture(video_input)
print("Capture video: %s" %(video_input))

success,image = vidcap.read()
count = 0
success = True
while success:
  frameName = "vid1_frame"
  if count < 10:
    frameName += "00{}.jpg".format(count)
  elif count < 100:
    frameName += "0{}.jpg".format(count)
  else:
    frameName += "{}.jpg".format(count)
  cv2.imwrite(frameName, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', str(success), 'count: ', count)
  count += 1
