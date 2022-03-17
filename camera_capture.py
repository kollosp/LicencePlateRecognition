import os
os.environ['DISPLAY'] = ':0'
import cv2
import numpy as np 
import random
import time
import psutil

start = time.time()
end = float('inf')
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,1280);
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,720);

click_param = {
    'en_contour': True,
    'en_canny': True,
    'pick_photo': False,
}

WIDTH, HEIGHT = 480, 320

def onclick(event, x, y, flags, param):
    if  event == cv2.EVENT_LBUTTONDOWN: 
        if x < WIDTH/2 and y < HEIGHT/2:
            exit()
        elif x > WIDTH/2 and y < HEIGHT/2:
            param['pick_photo'] = True
            pass
            #en_canny = not en_canny
        elif x < WIDTH/2 and y > HEIGHT/2:
            print("contour", WIDTH)
            param['en_contour'] = not param['en_contour']            
        else:
            param['en_canny'] = not param['en_canny']
  

cv2.namedWindow("test")
cv2.setMouseCallback("test", onclick, click_param)
cv2.setWindowProperty("test",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

last_saved_image = None

mat = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
frame_quarter_shape = (int(HEIGHT/2), int(WIDTH/2))

print("frame_quarter_shape", frame_quarter_shape)
fps_time = time.time()

img_counter = 0

while True:
    cascade = cv2.CascadeClassifier()
    cascade.load(cv2.samples.findFile("haar_cascade/cascade48x24.xml"))

    ret, original = cam.read()
    if not ret:
        print("failed to grab frame")
        break
 
    #frame = cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = original.copy()
    frame_resized = cv2.resize(frame,frame_quarter_shape)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    #30s interval
    end = time.time()
    if end - start > 10 or click_param['pick_photo']:
        click_param['pick_photo'] = False
        img_counter = img_counter+1
        timestr = time.strftime("%Y%m%d%H%M%S")
        img_name = "/home/pi/77-car-camera/data/{0}_{1}_camera_capture.png".format(timestr, img_counter)
        cv2.imwrite(img_name, original)
        print("{} written!".format(img_name))
        start = time.time()
        last_saved_image = frame_resized
        mat[int(frame.shape[1]/2):int(frame.shape[1]), int(frame.shape[0]/2):int(frame.shape[0])] = last_saved_image
   

    #show original frame
    mat[:int(frame.shape[1]/2), int(frame.shape[0]/2):int(frame.shape[0])]= frame_resized
    #show canny edges 
    if click_param['en_canny']:
        #haar cascade
        rects = cascade.detectMultiScale(original)

        for _, rect in enumerate(rects):

            pass
        #mat[int(frame.shape[1] / 2):int(frame.shape[1]), :int(frame.shape[0] / 2)] =


        #simple canny
        #mat[int(frame.shape[1]/2):int(frame.shape[1]), :int(frame.shape[0]/2)]= cv2.cvtColor(cv2.Canny(frame_resized,120,200),cv2.COLOR_GRAY2RGB)
        mat[int(frame.shape[1]/2):int(frame.shape[1]), :int(frame.shape[0]/2)]= cv2.cvtColor(cv2.Canny(frame_resized,150,200),cv2.COLOR_GRAY2RGB)


    #calculate contorus
    if click_param['en_contour']:
        canny_fullsize = cv2.Canny(frame,150,200)
        contours, hierarchy = cv2.findContours(canny_fullsize, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        con_image = frame.copy()
        
        #filtering contours
        contours_filtered = [x for x in contours if len(x) > 10 and len(x) < 100]    

        for index, contour in enumerate(contours_filtered):
            cv2.drawContours(con_image, [contour], -1, (0,0,random.randint(100,255)),5)
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(con_image,(x,y),(x+w,y+h),(0,255,0),2)

        #print("contours found, ", len(contours))
        mat[:int(frame.shape[1]/2), :int(frame.shape[0]/2)] = cv2.resize(con_image, frame_quarter_shape)
    
    #draw cross to distinguish all quarters
    cv2.line(mat, (int(mat.shape[1]/2), 0),  (int(mat.shape[1]/2), mat.shape[0]), (255,0,0)) 
    cv2.line(mat, (0, int(mat.shape[0]/2)),  (mat.shape[1], int(mat.shape[0]/2)), (255,0,0)) 
    mat_rot =  cv2.rotate(mat, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    #draw fps and contours count
    fps = 1.0 / (time.time() - fps_time)
    fps_time = time.time()
    next_pick_img = int(10 - (end-start))
    cpu = psutil.cpu_percent()
    cv2.putText(mat_rot, "{0:.0f} fps".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0),3)
    cv2.putText(mat_rot, "{0:.0f} fps".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),1)
    cv2.putText(mat_rot, "{} con.".format(len(contours)), (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0),3)
    cv2.putText(mat_rot, "{} con.".format(len(contours)), (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),1)
    cv2.putText(mat_rot, "{} imgs.".format(img_counter), (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0),3)
    cv2.putText(mat_rot, "{} imgs.".format(img_counter), (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),1)
    cv2.putText(mat_rot, "{}s next".format(next_pick_img), (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0),3)
    cv2.putText(mat_rot, "{}s next".format(next_pick_img), (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),1)
    cv2.putText(mat_rot, "{0:.0f}% cpu".format(cpu), (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0),3)
    cv2.putText(mat_rot, "{0:.0f}% cpu".format(cpu), (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),1)
    #display image
    #cv2.imshow("test",mat)
    #cv2.imshow("test", cv2.rotate(mat, cv2.ROTATE_90_CLOCKWISE))
    cv2.imshow("test",mat_rot)

    k = cv2.waitKey(1)
cam.release()
cv2.destroyAllWindows()
