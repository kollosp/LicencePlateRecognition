import os
os.environ['DISPLAY'] = ':0'
import cv2
import numpy as np 
import random
import time

start = time.time()
end = float('inf')
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")
cv2.setWindowProperty("test",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

WIDTH, HEIGHT = 480, 320

def f():
    exit()

cv2.createButton("Back",f,None,cv2.QT_PUSH_BUTTON,1)
img_counter = 0


while True:
    ret, original = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    
    frame = cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
   
    frame_quarter_shape = (int(frame.shape[1]/2), int(frame.shape[0]/2))

    mat = np.zeros(frame.shape, np.uint8)
    cv2.line(mat, (int(mat.shape[1]/2), 0),  (int(mat.shape[1]/2), mat.shape[0]), (255,0,0)) 
    cv2.line(mat, (0, int(mat.shape[0]/2)),  (mat.shape[1], int(mat.shape[0]/2)), (255,0,0)) 
    #frame = cv2.resize(frame, (int(WIDTH/2),int(HEIGHT/2)))

    frame_resized = cv2.resize(frame,frame_quarter_shape)

    #show original frame
    mat[:int(frame.shape[0]/2), :int(frame.shape[1]/2)] = frame_resized
    #show canny edges 
    mat[int(frame.shape[0]/2):int(frame.shape[0]), :int(frame.shape[1]/2)]= cv2.cvtColor(cv2.Canny(frame_resized,120,200),cv2.COLOR_GRAY2RGB)


    #calculate contorus
    canny_fullsize = cv2.Canny(frame,120,200)
    contours, hierarchy = cv2.findContours(canny_fullsize, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    con_image = frame.copy()
    
    #filtering contours
    contours_filtered = [x for x in contours if len(x) > 10 and len(x) < 100]    

    for index, contour in enumerate(contours_filtered):
        #print("f", len(contour))
        cv2.drawContours(con_image, [contour], -1, (0,0,random.randint(100,255)),5)
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(con_image,(x,y),(x+w,y+h),(0,255,0),2)

    #print("contours found, ", len(contours))
    mat[:int(frame.shape[0]/2), int(frame.shape[1]/2):int(frame.shape[1])] = cv2.resize(con_image, frame_quarter_shape)

    #cv2.imshow("test",frame_resized)
    cv2.imshow("test", mat)

    k = cv2.waitKey(1)

    #30s interval
    if end - start > 10:
        timestr = time.strftime("%Y%m%d%H%M%S")
        img_name = "/home/pi/camera/data/{}_camera_capture.png".format(timestr)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        start = time.time()

    end = time.time()

cam.release()
cv2.destroyAllWindows()
