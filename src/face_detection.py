import cv2
import webcolors


class FaceData:
    def __init__(self):
        self.face_ratio = None
        self.face_color = None
        self.nose_size = None
        self.nose_rating = None
        self.eye_color = None
        self.ear_size = None
        self.ear_rating = None

    def __repr__(self) -> str:
        return f"Face Ratio-{self.face_ratio}, Face Color-{self.face_color} Nose-{self.nose_rating}, Eye-{self.eye_color}, Ear-{self.ear_rating}"


class FaceDetection:

    @classmethod
    def detect(cls, image:str):
        face_data = FaceData()

        img = cv2.imread(image);                                                         #read image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                        #convert image to gray color

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')         #run cascade classifiers

        #FOR NOSE
        nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

        #FOR EYE
        eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

        #FOR EAR
        ear_cascade = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')

        #FOR HAIR
        # hair_cascade = cv2.CascadeClassifier('haarcascade_hair.xml')




        faces = face_cascade.detectMultiScale(gray,1.3,5)                                   #detect face using cascade classifier

        for (x, y, w, h) in faces:
            cv2.rectangle(img,(x, y), (x+w, y+h), (255, 255, 255), 2)                       #draw rectangle around the face

            ratio = w/h
            # print("Width to Height R  atio: ", ratio)                                          #calculate width to height ratio of the face


            face_gray = gray[y:y+h, x:x+w]                                                  #crop the face from the image
            face_color = img[y:y+h, x:x+w]


            avg_color = cv2.mean(face_color)[:3]                                            #extract only RGB colors
            hex_color = webcolors.rgb_to_hex(tuple(int(round(c)) for c in avg_color))       #convert the average color to HTML hex code
            cv2.putText(img, hex_color, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            face_data.face_color = avg_color
            face_data.face_ratio = ratio



        #FOR NOSE
            noses = nose_cascade.detectMultiScale(face_gray, 1.3, 5)                        #detect the nose in the face

            for (nx, ny, nw, nh) in noses:                                                  
                cv2.rectangle(face_color, (nx, ny), (nx+nw, ny+nh), (255, 0, 0), 1)         #draw rectangle around the nose

                nose_size = nw * nh                                                         #calculate nose size

                if nose_size < 2000:                                                        #rate the nose in to 3 types
                    nose_rating = 'Small'
                elif nose_size < 4000:
                    nose_rating = 'Medium'
                else:
                    nose_rating = 'Large'

                cv2.putText(face_color, ("Size: {nose_size}, Rating: {nose_rating}"),(nx, ny-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                face_data.nose_rating = nose_rating
                face_data.nose_size = nose_size
                                                                                            #output the nose size , rating



        #FOR EYE
            eyes = eye_cascade.detectMultiScale(face_gray, 1.3, 5)                          #detect the eyes in the face
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 1)         #draw rectangle around the eyes

                eye_roi = face_gray[ey:ey+eh, ex:ex+ew]

                eye_color = cv2.mean(eye_roi)[:3]  
                hex_color = webcolors.rgb_to_hex(tuple(int(round(c)) for c in eye_color))                                         #detect the eye color
                face_data.eye_color = eye_color

                cv2.putText(face_color, ("Eye color: {eye_color}"), (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                                                                                            #output the eye color




        #FOR EAR
            ear = ear_cascade.detectMultiScale(face_gray, 1.3, 5)  
                                    #detect the ear in the face

            for (ix, iy, iw, ih) in ear:                                                  
                cv2.rectangle(face_color, (ix, iy), (ix+iw, iy+ih), (255, 0, 0), 1)         #draw rectangle around the ea

                ear_size = iw * ih                                                          #calculate ear size
                ear_rating = 'Large'

                if ear_size < 200:                                                          #rate the ear in to 3 types
                    ear_rating = 'Small'
                elif ear_size < 400:
                    ear_rating = 'Medium'

                print(ear_rating)

                cv2.putText(face_color, ("Size: {ear_size}, Rating: {ear_rating}"),(ix, iy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                face_data.ear_size = ear_size
                face_data.ear_rating = ear_rating
                                                                                            #output the ear size , rating


        """
        #FOR HAIR
            hair = hair_cascade.detectMultiScale(face_gray, 1.3, 5)                         #detect the hair 
            
            for (hx, hy, hw, hh) in hair:

                hair_roi = face_gray[hy:hy+hh, hx:hx+hw]

                hair_color = cv2.mean(hair_roi)[:3]                                         #detect the hair color

                cv2.putText(face_color, ("Hair color: {hair_color}"), (hx, hy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                                                                                            #output the hair color        


        #FOR SKIN                                                                           #extract the skin region of interest
            skin_roi = cv2.cvtColor(face_color, cv2.COLOR_BGR2HSV)                          #convert it to HSV
            skin_roi = cv2.inRange(skin_roi, (0, 10, 60), (20, 150, 255))                   #apply a threshold
            skin_roi = cv2.GaussianBlur(skin_roi, (3,3), 0)                                 #blur the skin region

            skin_color = cv2.mean(face_color, mask = skin_roi)[:3]                          #detect the skin color

            cv2.putText(face_color, ("Skin color: {skin_color}"), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                                                                                            #output the skin color
        """

        return face_data
                                                                                    

# cv2.imshow("test", img);   
# print(face_data)                                                         #output the face detect image
# cv2.waitKey(0)
# cv2.destroyAllWindows()
