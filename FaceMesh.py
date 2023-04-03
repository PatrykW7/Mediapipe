import cv2
import mediapipe as mp
import time


cap=cv2.VideoCapture(0)
#previous time
pTime=0
#current time
cTime=0

mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
#Modyfikacja kontorow twarzy
drawSpec=mpDraw.DrawingSpec(thickness=1,circle_radius=1)


while True:
    success,img=cap.read()
    imageRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=faceMesh.process(imageRGB)

    if results.multi_face_landmarks:
        #faceLms to jest jedna twarz
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLms,mpFaceMesh.FACEMESH_CONTOURS,
                                    landmark_drawing_spec=drawSpec,connection_drawing_spec=drawSpec)
            #przejscie po punktach na twarzy
            #lm to sa punkty na jednej twarzy
            for id, lm in enumerate(faceLms.landmark):
                #konwersja punktow na wartosci pixelowe
                #ic = image chanels ?
                ih,iw,ic=img.shape
                x,y=int(lm.x*iw),int(lm.y*ih)
                print(id,x,y)
                






    cTime=time.time() 
    fps=1/(cTime-pTime)
    pTime=cTime
      

    cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    #cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)

    