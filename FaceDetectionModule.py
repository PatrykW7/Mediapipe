import cv2
import mediapipe as mp
import time



class FaceDetect():
    def __init__(self):
        #self.minDetectConf=minDetectConf


        self.mpDraw=mp.solutions.drawing_utils
        self.mpFace=mp.solutions.face_detection
        self.face=self.mpFace.FaceDetection()
        #self.face=self.mpFace.FaceDetection(self.minDetectConf)

    def findFaces(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.face.process(imgRGB)
        bboxs= []
        if self.results.detections:
            #bez enumerate nie dziala 
            for id,detection in enumerate(self.results.detections):
                bboxC=detection.location_data.relative_bounding_box           
                ih,iw,ic=img.shape
                bbox=int(bboxC.xmin*iw),int(bboxC.ymin*ih),\
                        int(bboxC.width*iw),int(bboxC.height*ih)

                if draw:
                    img =self.fancyDraw(img,bbox)
                
                bboxs.append([id,bbox,detection.score])
                
                
                #dodanie tekstu nad bbox (tym kwadratem na twarzy)
                cv2.putText(img,f'{int(detection.score[0]*100)}%',
                (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
        return img,bboxs
    
    def fancyDraw(self,img,bbox):
        #x,y to sa punkty poczatkowe
        l=30
        t=5
        rt=1
        x,y,w,h=bbox
        #x1 y2 to sa punkty po przekatnej (prawy dolny rog)
        x1,y1= x+w,y+h

        #narysowanie prostokatu
        cv2.rectangle(img,bbox,(255,0,255),rt)
        #Top left x,y
        cv2.line(img,(x,y),(x+l,y),(255,0,255),t )
        cv2.line(img,(x,y),(x,y+l),(255,0,255),t )
        #Top right x1,y
        cv2.line(img,(x1,y),(x1-l,y),(255,0,255),t )
        cv2.line(img,(x1,y),(x1,y+l),(255,0,255),t )
        #Bottom left x,y1
        cv2.line(img,(x,y1),(x+l,y1),(255,0,255),t )
        cv2.line(img,(x,y1),(x,y1-l),(255,0,255),t )
        #Bottom right x1,y1
        cv2.line(img,(x1,y1),(x1-l,y1),(255,0,255),t )
        cv2.line(img,(x1,y1),(x1,y1-l),(255,0,255),t )
        return img

        

def main():
    cap=cv2.VideoCapture(0)
    pTime=0
    cTime=0
    detector=FaceDetect()



    while True:
        success,img=cap.read()
        img,bboxs=detector.findFaces(img)
        
        #wyswietlenie informacji o bboxs
        #print(bboxs)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),3)
        cv2.imshow("Image",img)
        cv2.waitKey(1)
        




if __name__ == "__main__":
    main()