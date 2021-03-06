import cv2
# face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')

# 调用摄像头
capture = cv2.VideoCapture(0)

while(True):
    
    # capture the image of came
    flag, frame = capture.read()
    if not flag:
        break
    faces = face_cascade.detectMultiScale(frame, 1.3, 2)
    img = frame
    for (x, y, w, h) in faces:
        # plot the face area, blue
        img = cv2.rectangle(img,(x,y),(x+w, y+h), (255, 0, 0), 2)

        face_area = img[y:y+h, x:x+w]

        # eyes
        eyes = eye_cascade.detectMultiScale(face_area, 1.3, 10)
        for (ex, ey, ew, eh) in eyes:
            # bgr
            cv2.rectangle(face_area, (ex, ey),(ex+ew, ey+eh),(0,255,0),1)

        smiles = smile_cascade.detectMultiScale(face_area,scaleFactor= 1.16,minNeighbors=65,minSize=(25, 25),flags=cv2.CASCADE_SCALE_IMAGE)
        for (ex,ey,ew,eh) in smiles:
            cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,0,255),1)
            cv2.putText(img,'Smile',(x,y-7), 3, 1.2, (0, 0, 255), 2, cv2.LINE_AA)


    cv2.imshow('frame2',img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
# close all the windows
capture.release()
cv2.destroyAllWindows()