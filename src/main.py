import os
import uuid
from flask import Flask, request, render_template

from glass import eys
from face_detection import FaceDetection
from gender_detection import GenderDetection
from sunglass_modal import match_glass

UPLOAD_FOLDER = './static/upload'

app = Flask(__name__,static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'This is your secret key to utilize session in Flask'


def add_sunglass(image):
    gender = GenderDetection.detect(image)
    face_data = FaceDetection.detect(image)

    face_ratio = face_data.face_ratio
    face_color = face_data.face_color
    nose = face_data.nose_rating
    eye_color = face_data.eye_color

    matched_glass = match_glass(gender,face_ratio,eye_color,nose)

    path = eys.add_glass_to_image(image,'static/sunglasses/square-black-green.png')

    return matched_glass, path


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4())+'.jpg')
        file1.save(path)
        print(path)
        sunglass_data, image_path = add_sunglass(path)
        return render_template('show_image.html', image=image_path, frame=sunglass_data[1], frame_color=sunglass_data[0], glass_color=sunglass_data[2])
    return render_template('index.html')


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()

