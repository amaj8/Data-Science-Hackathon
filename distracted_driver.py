import cv2
import numpy as np
import os

SZ=20
bin_n = 16 # Number of bins

svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR,
                    svm_type = cv2.ml.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

####------- get the HOG features
def hog(img):
    try:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
        bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)     # hist is a 64 bit vector
        return hist

    except Exception as e:
        print(e)
        return None


driver_list = open('new_driver_list','r')

train_img_home = '/home/alpha/driver_imgs/train/'
test_img_home ='/home/alpha/driver_imgs/test/'

train_imgs = []
train_labels = []

lines = driver_list.readlines()
# hdr = driver_list.readline()
half = len(lines)/2
train = lines[1:half]
test = lines[half:len(lines)]

for l in train:
    _,classname,imgname = l.strip().split(',')
    img_path = train_img_home+classname+'/'+imgname
    img = cv2.imread(img_path,0)
    h = hog(img)
    # if h is None:
    #     os.remove(img_path)
    # else:
    #     new_driver_list.write(l)
    train_imgs.append(h)

    classname = ord( classname[-1] ) - ord('0')
    train_labels.append(classname)

train_data = np.float32(train_imgs)
train_labels = np.int32(train_labels)

#####This one is correct!!
# test_img_path = test_img_home + 'img_11.jpg'


##### ------- Setting the SVM to train
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(train_data,cv2.ml.ROW_SAMPLE,train_labels)
svm.save('svm_data2.dat')

######--------Testing
test_img_path = test_img_home + 'img_11.jpg'
test_img = cv2.imread(test_img_path,0)

correct_response = []
result = []

for l in test:
    _, classname, imgname = l.strip().split(',')
    img_path = train_img_home + classname + '/' + imgname
    img = cv2.imread(img_path, 0)
    hog_test_data = hog(test_img)
    test_data = np.float32(hog_test_data)
    test_data = test_data.reshape(1, test_data.shape[0])
    result.append(svm.predict(test_data)[1])
    classname = ord(classname[-1]) - ord('0')
    correct_response.append(classname)

# hog_test_data = hog(test_img)
# test_data = np.float32(hog_test_data)
# test_data = test_data.reshape(1,test_data.shape[0])
# result = svm.predict(test_data)
# print result

#######   Check Accuracy   ########################
mask = result==correct_response
correct = np.count_nonzero(mask)
print correct*100.0/len(result)

