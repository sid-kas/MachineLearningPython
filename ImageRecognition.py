import numpy as np
import cv2
import urllib.request
import os




def store_raw_images(link,directory):
    image_urls = urllib.request.urlopen(link).read().decode()

    if not os.path.exists(directory):
        os.makedirs(directory)
    pic_num = 1

    for i in image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, directory + '/' + str(pic_num) + '.jpg')
            img = cv2.imread(directory+'/'+str(pic_num) + '.jpg', cv2.IMREAD_COLOR)
            resized_image = cv2.resize(img, (100,100))
            cv2.imwrite(directory+'/'+str(pic_num) + '.jpg', resized_image)
            pic_num += 1

        except Exception as e:
            print(str(e))



link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07687626'
directory = 'bread'
store_raw_images(link,directory)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    cv2.imshow('frame',frame)
    cv2.imshow('gray', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()