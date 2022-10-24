import os
import cv2
import time

# 1.Top
# 2.Bottom
# 3.Right
# 4.Left
# 5.Stop
# 6.Attack

path_data = "train"
label = "Top"

cap = cv2.VideoCapture(0)
index_image = 1
index = index_image
while True:
    _, frame = cap.read()

    print(index_image)

    frame = cv2.resize(frame, (600, 600))
    frame = cv2.flip(frame, 1)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame = cv2.bilateralFilter(frame, 5, 20, 20)

    if not os.path.exists(path_data):
        os.mkdir(path_data)
    if not os.path.exists(os.path.join(path_data, label)):
        os.mkdir(os.path.join(path_data, label))

    name = label + "_" + str(index_image)
    name_image = name + ".jpg"

    x = 600 - 400
    y = 0
    h = 400
    w = 400

    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow("anh", frame)

    img_hand = frame[0:400, 200:600]

    img_resize = cv2.resize(img_hand, (240, 240))
    cv2.imwrite(os.path.join(path_data, label, name_image), img_resize)

    index_image += 1
    if index_image % 1601 == 0:   # lưu tới ảnh n-1
        break

    k = cv2.waitKey(200)
    if k == ord('x'):
        break


cap.release()
cv2.destroyAllWindows()
