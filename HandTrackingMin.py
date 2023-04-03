import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils


pTime=0
cTime=0


while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    #wyswietla wspolrzedne dloni, jesli widac je na kamerze
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #wyswietlanie wspolrzednych kazdego punktu na dloni
            for id,lm in enumerate(handLms.landmark):
               # print(id,lm)
                #zmiana wartosci wspolrzednych na pixele
                h, w, c=img.shape              
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                #narysowanie kolka o danym kolorze przy nadgarstku
                #if id==4:
                   # cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)

            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

    #Tworzenie Frame rate
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    #wyswietlanie Frame rate
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)


    cv2.imshow("Image",img)
    cv2.waitKey(1)



