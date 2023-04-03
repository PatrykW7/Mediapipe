import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
pTime=0
cTime=0

mpFace=mp.solutions.face_detection
face=mpFace.FaceDetection(0.75)
mpDraw=mp.solutions.drawing_utils



while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=face.process(imgRGB)
    #results.detections to sa wspolrzedne 
    #print(results.detections)
    #multiple faces
    if results.detections:
        #bez enumerate nie dziala 
        for id,detection in enumerate(results.detections):
            #id odnosi sie do numeru twarzy (jakby bylo wiecej niz jedna)
            #print(id,detection,detection.score)
            #print(detection.score)
            #relative_bounding_box t jest xmin, ymin, width, height
            #print(detection.location_data.relative_bounding_box)
            #Twarz w kwadracie (bounding box), z kilkoma punktami
            #domyslne rysowanie, linijka ponizej
            #mpDraw.draw_detection(img,detection)
            #zmiana na pixele
            #rysowanie customowe
            #print(detection.location_data.relative_bounding_box)
            bboxC=detection.location_data.relative_bounding_box           
            ih,iw,ic=img.shape
            bbox=int(bboxC.xmin*iw),int(bboxC.ymin*ih),\
                    int(bboxC.width*iw),int(bboxC.height*ih)
            cv2.rectangle(img,bbox,(255,0,255),2)
            #dodanie tekstu nad bbox (tym kwadratem na twarzy)
            cv2.putText(img,f':{int(detection.score[0]*100)}%',
            (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)




    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
