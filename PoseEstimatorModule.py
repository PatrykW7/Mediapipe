import cv2
import mediapipe as mp
import time

class bodyDetector():
    def __init__(self,mode=False,model_complexity=1,smooth=True,ensegm=True,smoth_segm=True,detectionCon=0.5,trackCon=0.5):
       
        self.mode=mode
        self.model_complexity=model_complexity   
        self.smooth=smooth
        self.ensegm=ensegm
        self.smoth_segm=smoth_segm
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose       
        self.pose=self.mpPose.Pose(self.mode,self.model_complexity,self.smooth,self.ensegm,self.smoth_segm,self.detectionCon,self.trackCon)
        
        


    def findBody(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #
        #print(dir(self))
        self.results = self.pose.process(imgRGB)    
        if self.results.pose_landmarks:
            #wypisywanie wspolrzednych nosa 
            #print(self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.NOSE].x * 360)
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
                

        return img

    def getPosition(self,img,draw=True):
        lmList=[]
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                #print(lmList[id][0])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        
        return lmList

    def make_1080(self,cap):
        cap.set(3,1920)
        cap.set(4,1080)
        return cap




def main():
    cap=cv2.VideoCapture(0)
    pTime=0
    detector=bodyDetector()
    cap=detector.make_1080(cap)
    #lmList=detector.getPosition(img)
    

    while True:
        success,img=cap.read()
        img=detector.findBody(img)
        lmList=detector.getPosition(img,draw=False)
        #wyswietlenie okreslonego landmarka
        #if len(lmList)!=0:
        #print(lmList[5])
            #rysowanie w okreslony sposob wybranego punktu (np. 14)
            #jak przestaje wykrywac punkt, to sam sie program wylacza
            #cv2.circle(img,(lmList[14][1],lmList[14][2]),15,(0,0,255),cv2.FILLED)
        cTime=time.time() 
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        

        cv2.imshow("Image",img)
        cv2.waitKey(1)
#       cap.release()
#cv2.destroyAllWindows

if __name__ == "__main__":
    main()