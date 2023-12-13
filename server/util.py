import cv2
import joblib
import numpy as np
import base64
import json
from wavelet import w2d

__class_name_to_number={}
__class_number_to_name={}

__model=None

def classify_image(img_b64,file_path=None):
    global __class_name_to_number
    global __model
    global __class_number_to_name
    imgs=get_if_2_eyes(file_path,img_b64)

    result=[]
    for img in imgs:
        scaled_raw_img=cv2.resize(img,(32,32))
        im_haar=w2d(img,'db1',5)
        scaled_img_haar=cv2.resize(im_haar,(32,32))
        combined_img=np.vstack((scaled_raw_img.reshape(32*32*3,1),scaled_img_haar.reshape(32*32,1)))
        len_img_arr=32*32*3+32*32
        final=combined_img.reshape(1,len_img_arr).astype(float)

        result.append({
            "class":class_no_to_name(__model.predict(final)[0]),
            "class_prob":np.round(__model.predict_proba(final)*100,2).tolist()[0],
            "class_dictionary":__class_name_to_number
        })
        
    return result
def class_no_to_name(num):
    return __class_number_to_name[num]
def get_cv2_image_from_b64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img




def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dict.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")


def get_if_2_eyes(img_path,img_b64_data):
    face_cascade=cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_frontalface_default.xml")
    eye_cascade=cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_eye.xml")
    if img_path:
        print("nee engada inga")
        img=cv2.imread(img_path)
    else:
        img=get_cv2_image_from_b64_string(img_b64_data)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    cropped_faces=[]
    for (x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        if len(eyes)>=2:
            cropped_faces.append(roi_color)
    return cropped_faces
def get_b64_for_virat():
    with open("b64.txt") as f:
        return f.read()
    
if __name__=="__main__":
    load_saved_artifacts()
    #print(classify_image(get_b64_for_virat(),None))
    print(classify_image(None,"./test_images/virat3.jpg"))

