### 还没改好

import torch
import cv2 as cv
import numpy as np

def video_landmark_demo():
    cnn_model = torch.load("./age_gender_model.pth")
    print(cnn_model)
    # capture = cv.VideoCapture(0)
    capture = cv.VideoCapture("D:/images/video/example_dsh.mp4")

    # load tensorflow model
    net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)
    while True:
        ret, frame = capture.read()
        if ret is not True:
            break
        frame = cv.flip(frame, 1)
        h, w, c = frame.shape
        blobImage = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        net.setInput(blobImage)
        cvOut = net.forward()
        # 绘制检测矩形
        for detection in cvOut[0,0,:,:]:
            score = float(detection[2])
            if score > 0.5:
                left = detection[3]*w
                top = detection[4]*h
                right = detection[5]*w
                bottom = detection[6]*h

                # roi and detect landmark
                roi = frame[np.int32(top):np.int32(bottom),np.int32(left):np.int32(right),:]
                rw = right - left
                rh = bottom - top
                img = cv.resize(roi, (64, 64))
                img = (np.float32(img) / 255.0 - 0.5) / 0.5
                img = img.transpose((2, 0, 1))
                x_input = torch.from_numpy(img).view(1, 3, 64, 64)
                age_, gender_ = cnn_model(x_input.cuda())
                predict_gender = torch.max(gender_, 1)[1].cpu().detach().numpy()[0]
                gender = "Male"
                if predict_gender == 1:
                    gender = "Female"
                predict_age = age_.cpu().detach().numpy()*116.0
                print(predict_gender, predict_age)

                # 绘制
                cv.putText(frame, ("gender: %s, age:%d"%(gender, int(predict_age[0][0]))), (int(left), int(top)-15), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
                cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
               # cv.putText(frame, "score:%.2f"%score, (int(left), int(top)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                c = cv.waitKey(10)
                if c == 27:
                    break
                cv.imshow("face detection + landmark", frame)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    video_landmark_demo()