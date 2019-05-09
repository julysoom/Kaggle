# Screw-Defect-Detection
* Classify whether a screw is defected or not based on an image data set of screws.
* Data provided by a company that manages integrated information management system
* Company name masked due to contract issues
## Requirement
* Python 3.5 (or later)
* keras
* convolutional 
* cv2
* glob
## Methods for Training and Prediction
* ``cv2 ``
* ``Convolutional Neural Network``
## How to use
### Data
* Used following user defined function to import train data
* Data contains both defected and normal screw images
* Codes will be explained using annotations
* ```cropped_nasa_bad``` code to import defected screw images
```python
def cropped_nasa_bad(folder_name, make_folder_name) :
    def make_file_path_data_frame(path) : 
        file_list = os.listdir(path) # setting file directory
        file_list.sort() # sorting image files in order
        file_data_frame = pd.DataFrame(file_list, columns = ["file_path"]) # making a dataframe of file lists
        if path[-1] == "/" : ## if file contains "/" set following path as below 
            file_data_frame = path + file_data_frame["file_path"]
        else : # if file doesn't contain "/", follow code below
            file_data_frame = path + "/" + file_data_frame["file_path"]
        file_data_frame = pd.DataFrame(file_data_frame)
        return file_data_frame
    
    file_names = make_file_path_data_frame("C://Users/irie9/Python/" + folder_name) # setting file names
    
    for j in range(len(file_names["file_path"])) : # cropping and intensifying image by using cv2 for better detection
        img = cv2.imread(file_names["file_path"][j], 0)
        ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        titles = "BINARY"
        img_t = thresh
        img = cv2.medianBlur(img_t, 5)
        cimg = cv2.cvtColor(img_t, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, img.shape[1], # detecting screw as a circle
                                  param1=200, param2=20, minRadius=300,
                                  maxRadius=350)
        W = int(circles[0, 0, 0]) 
        H = int(circles[0, 0, 1])
        r = int(circles[0, 0, 2])
        circles = np.uint16(np.around(circles))
        
        for i in circles[0, :] : # calculating the optimized cropping size for the image
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        cropped_img = img_t[H-r-20:H+r+20, W-r-20:W+r+20] 
        path = "C://Users/irie9/Python/" + make_folder_name + "/"
        cv2.imwrite(path + "bad_" + str(j) + ".png", cropped_img)
* Using ```cropped_nasa_bad``` code as following
```python      
cropped_nasa_bad("bad", "c_bad") # "bad" is the path to refer to activate the code while "c_bad" is the new folder name that will be generated
* Same applies for ```cropped_nasa_good```
* Sample image imported


![image](https://user-images.githubusercontent.com/42960718/53015495-9c12f800-348e-11e9-9f8c-ef9d50ddcfcc.png)
### Import Data
```python

import glob
images = glob.glob("C://Users/irie9/Python/c_nasa/*.png")
X = []
for i in images :
    X.append((cv2.imread(i, 0)))
norm = []
for i in range(len(X)) :
    norm.append(cv2.resize(X[i], (320, 320), cv2.INTER_AREA)/255)
X = norm
```
```python
path = "C://Users/irie9/Python/c_nasa"
labels = []
file_list = os.listdir(path)
for item in file_list :
    if item.find("good_") : # setting normal screw as "0"
        labels.append(0)
    elif item.find("bad_") : # setting defected screw as "1"
        labels.append(1)
    else:
        labels.append(-1) # setting undefined screw as "-1"
```
### Data Pre-processing
* Creating test y 
```python
nb_classes = int(np.array(labels).max()+1)
nb_classes
y = np_utils.to_categorical(labels, nb_classes)
```
* Splitting data to train and test
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.array(X), y, test_size=.2, random_state=42)

X_train = X_train[:, :, :, np.newaxis]
X_train.shape
X_test = X_test[:, :, :, np.newaxis]
```
### Model
* Parameters
```python
IMG_CHANNELS = 1 # gray image
IMG_ROWS = 320 # size of image
IMG_COLS = 320

BATCH_SIZE = 32
NB_EPOCH = 20
NB_CLASSES = 2 # 0, 1
VERBOSE = 1
VALIDATION_SPLIT = .2
OPTIM = Adam(lr=.00001)
```
* Model
```python
model = Sequential()

model.add(Conv2D(1, (3, 3), padding="same", input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation("elu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.25))

model.add(Conv2D(10, (3, 3), padding="same"))
model.add(Activation("elu"))
model.add(Conv2D(10, (3, 3)))
model.add(Activation("elu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(512, kernel_initializer="he_normal"))
model.add(Activation("elu"))
model.add(Dropout(.5))

model.add(Dense(NB_CLASSES))
model.add(Activation("softmax"))
keras.regularizers.l1_l2(l1=.01, l2=.01)

model.summary()
```


![image](https://user-images.githubusercontent.com/42960718/53015908-82be7b80-348f-11e9-853c-e95bca751e36.png)
### Evaluation
```python

model.compile(loss="categorical_crossentropy", optimizer=OPTIM,
             metrics=["accuracy"])
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE,
                   epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT,
                   verbose=VERBOSE)
print("Testing...")
score = model.evaluate(X_test, y_test,
                      batch_size=BATCH_SIZE, verbose=VERBOSE)
print("\nTest score:", score[0])  
print("Test accuracy:", score[1])
```
### Performance
* Test accuracy : 0.832167829666938


![image](https://user-images.githubusercontent.com/42960718/53015980-bb5e5500-348f-11e9-9e4b-290dd060ea26.png)


