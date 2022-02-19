import cv2
from tracker import *

total=0


tracker=EuclideanDistTracker()
cap = cv2.VideoCapture("highway.mp4")
object_detector=cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)

while True:
    ret,frame=cap.read()
    height,width,_=frame.shape

    roi=frame[height//2:height,width//4:(width//4)*3]
    mask=object_detector.apply(roi)
    _,mask=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    contours,_= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detections=[]

    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>150 and area<300:
            # cv2.drawContours(frame,[cnt],-1,(0,255,0),2)
            x,y,w,h=cv2.boundingRect(cnt)
            
            detections.append([x,y,w,h])


    boxes_ids=tracker.update(detections)
    for box in boxes_ids:
        x,y,w,h,id=box
        cv2.putText(roi,str(id),(x,y-15),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
        total=str(id)



    cv2.imshow("Roi",roi)
    cv2.imshow("Frame",frame)
    cv2.imshow("Mask",mask)
    key=cv2.waitKey(30)
    if key==27:
        print(total)
        break
cap.release()
cv2.destroyAllWindows()