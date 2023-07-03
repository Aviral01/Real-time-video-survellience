from keras.models import load_model
import cv2
import numpy as np
import winsound
np.set_printoptions(suppress=True)

model = load_model("trained_Model.h5", compile=False)

class_names = open("labels.txt", "r").readlines()
  
camera = cv2.VideoCapture(0)

while True:
    ret, image = camera.read()

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    image_copy = image.copy()
    image_copy = cv2.putText(image_copy, "", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)

    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    image = (image / 127.5) - 1

    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    
    label = "{}".format(class_name[2:])

    image_copy = cv2.putText(image_copy, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Output", image_copy) 
    if (class_name.strip()!="7 Normal"):
        frequency = 2500  
        duration = 1000  
        winsound.Beep(frequency, duration)    
    keyboard_input = cv2.waitKey(1)

    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()

