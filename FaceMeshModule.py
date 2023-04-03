import cv2
import mediapipe as mp
import time

class FaceMeshDetector():


    def __init__(self,max_num_faces=2):
        self.max_num_faces=max_num_faces

        self.mpDraw=mp.solutions.drawing_utils
        self.mpFaceMesh=mp.solutions.face_mesh
        self.faceMesh=self.mpFaceMesh.FaceMesh(self.max_num_faces)    
        self.drawSpec=self.mpDraw.DrawingSpec(thickness=1,circle_radius=1)

    def findFaceMesh(self,img,draw=True):
        imageRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results=self.faceMesh.process(imageRGB)
        #obsluzenie wielu twarzy i dodanie ich wspolrzednych i id do listy 
        faces=[]
        if results.multi_face_landmarks:
            
            #faceLms to jest jedna twarz
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACEMESH_CONTOURS,
                                            landmark_drawing_spec=self.drawSpec,connection_drawing_spec=self.drawSpec)
                    #przejscie po punktach na twarzy
                    #lm to sa punkty na jednej twarzy
                    face=[]
                    for id, lm in enumerate(faceLms.landmark):
                        #konwersja punktow na wartosci pixelowe
                        #ic = image chanels ?
                        ih,iw,ic=img.shape
                        x,y=int(lm.x*iw),int(lm.y*ih)
                        #print(id,x,y)
                        #wyswietlenie kazdego punktu id na twarzy !
                        cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,0.5,(0,255,0),3)
                        face.append([x,y])
                        
                faces.append(face)
                
        return img,faces


    

def main():
    cap=cv2.VideoCapture(0)
    pTime=0
    cTime=0
    detector=FaceMeshDetector()

   


    while True:
        success,img=cap.read()
        img,faces=detector.findFaceMesh(img)
        #wypisanie wszystkich punktow
        #if len(faces)!=0:
        #    print(faces[0])
        cTime=time.time() 
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()