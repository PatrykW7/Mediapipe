import cv2
import time
#import klasy z innego pliku
import PoseEstimatorModule as pm
cap=cv2.VideoCapture(0)
pTime=0
#dziedziczenie z innego pliku
detector=pm.bodyDetector()
   
    

while True:
    success,img=cap.read()
    img=detector.findBody(img)
    lmList=detector.getPosition(img,draw=False)
    #wyswietlenie okreslonego landmarka
    if len(lmList)!=0:
        print(lmList[14])
        #rysowanie w okreslony sposob wybranego punktu (np. 14)
        #jak przestaje wykrywac punkt, to sam sie program wylacza
        cv2.circle(img,(lmList[14][1],lmList[14][2]),15,(0,0,255),cv2.FILLED)
    cTime=time.time() 
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)