import cv2
import numpy as np

class GenderDetection:

    @classmethod
    def detect(cls, image:str):

        # Load the pre-trained models for age and gender prediction
        age_net = cv2.dnn.readNetFromCaffe('Data/age/deploy1.prototxt', 'age_net.caffemodel')
        gender_net = cv2.dnn.readNetFromCaffe('Data/gender/deploy1.prototxt', 'gender_net.caffemodel')

        # Load the input image
        image = cv2.imread(image)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Loop over the detected faces
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face = image[y:y+h, x:x+w]

            # Preprocess the face ROI for age and gender prediction
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # Predict the age of the face
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = int(np.floor(np.sum(np.arange(0, 101) * np.transpose(age_preds))) / 200.0)

            # Predict the gender of the face
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = 'Male' if gender_preds[0][0] > gender_preds[0][1] else 'Female'

            # print(gender, age)
            return gender

            # Draw the predicted age and gender on the face ROI
            # label = '{}, {}'.format(gender, age)
            # cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)

# Display the output image
# cv2.imshow('Output', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



