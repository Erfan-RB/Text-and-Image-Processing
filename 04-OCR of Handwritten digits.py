#OCR of Handwritten digits
#OCR = Optical Character Recognition
import numpy as np
import cv2

image = cv2.imread('digits.png')
gray_img = cv2.cvtColor(image,
                        cv2.COLOR_BGR2BRAY)
divisions = list(np.hsplit(i,100) for i in np.vsplit(gray_img,50))
NP_array = np.array(divisions)
train_data = NP_array[:,:50].reshape(-1,400).astype(np.float32)
test_data = NP_array[:,50:100].reshape(-1,400).astype(np.float32) 

k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newacis]
test_labels  = np.repeat(k,250)[:,np.newaxis]

knn = cv2.ml.KNearest_create()
knn.train(train_data,
          cv2.ml.ROW_SAMPLE,
          train_labels)

ret, output ,neighbours,distance= knn.findNearest(test_data, k = 3)
matched = output==test_labels 
correct_OP = np.count_nonzero(matched) 

accuracy = (correct_OP*100.0)/(output.size)
print(accuracy)
