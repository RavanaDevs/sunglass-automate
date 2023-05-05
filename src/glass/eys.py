import cv2
import uuid
import os

def add_glass_to_image(face, glass):
    # Load the face and eyeglass images
    face_image = cv2.imread(face)
    eyeglass_image = cv2.imread(glass, -1)


    # resize input photo
    #img = cv2.resize( face_image, )
    #img = cv2.resize( face_image, )


    # Convert the face image to grayscale and detect faces
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Detect eyes in the face region
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = face_image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        # Draw bounding boxes around the detected eyes
        #for (ex, ey, ew, eh) in eyes:
        # cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Replace the area where the face was detected with the eyeglass image
        resized_eyeglass = cv2.resize(eyeglass_image, (w, h))
        mask = resized_eyeglass[:, :, 3]
        mask_inv = cv2.bitwise_not(mask)
        resized_eyeglass_bgr = resized_eyeglass[:, :, :3]
        face_roi = face_image[y:y+h, x:x+w]
        bg_roi = cv2.bitwise_and(face_roi, face_roi, mask=mask_inv)
        fg_roi = cv2.bitwise_and(resized_eyeglass_bgr, resized_eyeglass_bgr, mask=mask)
        combined_roi = cv2.add(bg_roi, fg_roi)
        face_image[y:y+h, x:x+w] = combined_roi

    image_path = os.path.join('static','edited', str(uuid.uuid4())+'.jpg')
    cv2.imwrite(image_path, face_image)
    return image_path
