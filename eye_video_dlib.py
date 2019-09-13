import numpy as np
import matplotlib.pyplot as plt
import dlib
import os
import cv2
from scipy.spatial import distance as dist
import json

def eye_video_dlib(video_path,save_path,threshold = 0.27,frame_detection = float("inf"),save_pic_or_not = None):

    video_name = os.path.basename(video_path)[:-4]
    save_path  = os.path.join(save_path, video_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_picture = save_path +"/picture"
    if not os.path.exists(save_picture):
        os.makedirs(save_picture)
    save_txt = save_path +"/txt"
    if not os.path.exists(save_txt):
        os.makedirs(save_txt)
        print("make_path")

    leftx_txt  = open(save_txt + "/left_eye_x.txt", 'a+')
    lefty_txt  = open(save_txt + "/left_eye_y.txt", 'a+')
    rightx_txt = open(save_txt + "/right_eye_x.txt", 'a+')
    righty_txt = open(save_txt + "/right_eye_y.txt", 'a+')
    aspect_ratio = open(save_txt+"/aspect_ratio.txt","a+")

    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    Y = [] #眼睛纵横比
    zero = 0
    data_ = []
    all_data = {}
    if cap.isOpened():
        success = True

    else:
        success = False
        print("读取失败!")
        return None

    while (success and frame_detection>0):
        frame_detection = frame_detection - 1
        success, frame = cap.read()
        if  not success:
            break
        """
        frame = np.rot90(frame, -1)
        frame = np.rot90(frame, -1)
        """
        detector = dlib.get_frontal_face_detector()
        # dlib的68点模型，使用作者训练好的特征预测器
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # 特征提取器的实例化
        dets = detector(frame, 1)
        print("人脸数：", len(dets))

        """
        cv2.imshow("test",frame)
        cv2.waitKey(0)
        
        if(len(dets)==0):
            if zero>float("inf"):
                break
            else:
                for i in range(3):
                    frame = np.rot90(frame,-1)
                    if len(detector(frame,1)) == 0:
                        continue
                    else:
                        print("旋转",i+1,"次后检测到人脸")
                        zero = 0
                        break
                dets = detector(frame, 1)"""
        if len(dets)==0:
            print("旋转3次后仍检测失败")
            cv2.imwrite(save_picture + "/{}wrong.jpg".format(zero), frame)
            zero += 1
            continue
        elif(len(dets)==1):
            frame_index += 1
            for k ,d in enumerate(dets):
                print("第", frame_index, "个人脸")
                width = d.right() - d.left()
                heigth = d.bottom() - d.top()
                shape = predictor(frame, d)
                print('人脸面积为：', (width * heigth))
                if save_pic_or_not:
                    # frame = frame[..., ::-1]
                    for i in range(68):
                        print(i,(shape.part(i).x, shape.part(i).y))
                        frame = cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), 1, 3)
                        # cv2.putText(frame, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255))

                    cv2.imwrite(save_picture + "/{}.jpg".format(frame_index), frame)


                left_x  = []
                left_y  = []
                right_x = []
                right_y = []
                pose = []
                score = [1,1,1,1,1,1,1,1,1,1,1,1]
                skeleton = {}

                for j in range(36, 42):
                    leftx_txt.write(str(shape.part(j).x) + " ")
                    left_x.append(shape.part(j).x)
                    pose.append(shape.part(j).x)
                leftx_txt.write("\n")
                for j in range(36, 42):
                    lefty_txt.write(str(shape.part(j).y) + " ")
                    left_y.append(shape.part(j).y)
                    pose.append(shape.part(j).y)
                lefty_txt.write("\n")
                for j in range(42, 48):
                    rightx_txt.write(str(shape.part(j).x) + " ")
                    right_x.append(shape.part(j).x)
                    pose.append(shape.part(j).x)
                rightx_txt.write("\n")
                for j in range(42, 48):
                    righty_txt.write(str(shape.part(j).y) + " ")
                    right_y.append(shape.part(j).y)
                    pose.append(shape.part(j).y)
                righty_txt.write("\n")

                skeleton["pose"] = pose
                skeleton["score"] = score
                skeleton_ = []
                skeleton_.append(skeleton)
                data = {}
                data["frame_index"] = frame_index
                data["skeleton"] = skeleton_
                data_.append(data)




                A = dist.euclidean((shape.part(37).x,shape.part(37).y), (shape.part(41).x,shape.part(41).y))
                B = dist.euclidean((shape.part(38).x,shape.part(38).y), (shape.part(40).x,shape.part(40).y))
                C = dist.euclidean((shape.part(36).x,shape.part(36).y), (shape.part(39).x,shape.part(39).y))
                l = (A + B) / (2.0 * C)
                A = dist.euclidean((shape.part(43).x,shape.part(43).y), (shape.part(47).x,shape.part(47).y))
                B = dist.euclidean((shape.part(44).x,shape.part(44).y), (shape.part(46).x,shape.part(46).y))
                C = dist.euclidean((shape.part(42).x,shape.part(42).y), (shape.part(45).x,shape.part(45).y))
                r = (A+B)/(2.0*C)
                Y.append((l+r)/2.0)
                aspect_ratio.write(str((l+r)/2.0))
                aspect_ratio.write("\n")
    all_data["data"] = data_
    all_data["label"] = str(video_path.split("/")[-2])
    all_data["label_index"] =int(video_path.split("/")[-2])
    with open(save_path+"/"
              +str(video_path.split("/")[-2])
              +"-"+str(video_path.split("/")[-1][:-4])
              +".json", "w") as f:
        json.dump(all_data, f)
    X = []
    Z = []
    blink = []
    count = 0
    total = 0
    lens = len(Y)
    for i in range(lens):
        X.append(i + 1)
        Z.append(threshold)
    for i in range(lens):
        if Y[i] < threshold:
            count += 1
        else:
            if count > 2:
                total += 1
            count = 0
        blink.append(total)

    plt.plot(X, Y, label="{}blink{}次".format(video_name, blink[-1]))
    plt.plot(X, Z)
    plt.legend()
    plt.savefig(save_path+"/{}.jpg".format(video_name))
    print("blink: ", blink[-1])
    plt.show()

# eye_video_dlib("D:/data_for_net/data-for-test/0-1.mp4","D:/data_for_net/data-for-test/",save_pic_or_not = True)
"""
f = cv2.imread("C:\\Users\\admin\\Desktop\\girl.png",cv2.IMREAD_GRAYSCALE)
f = f[350:420,300:420]
print(f.shape)
f = f[f>100]
print(f.shape)
cv2.imshow("f",f)
cv2.waitKey(0)
print(f)
"""








