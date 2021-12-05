from flask import Flask, request, render_template
from prediction import Predictor
import os
from flask import Flask, flash, request, redirect, url_for, send_file

from werkzeug.utils import secure_filename
from prediction import doPrediction

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# this is the python backend
# create the web app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global p
p=Predictor()

# routes are the endpoints
# they are the urls associated with the website
# url name is in the parentheses
# the function is called when the url is loaded
@app.route('/')
def index():
    print('index.html')
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/predictimage', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            global p
            #img = doPrediction(p, filename)
            img="upload/prediction/sarah_connor_grayscale.jpg_out.PNG"
            image = get_encoded_img(img)  # I use Linux path with `/` instead of `\`
    
            return jsonify({'image_url': image})
            #return send_file(img, mimetype='image/gif')
            # return redirect(url_for('prediction.html', filename=filename))
    return render_template('upload.html')

from flask import Flask, jsonify
from PIL import Image
import base64
import io

@app.route('/getImage') # as default it uses `methods=['GET']`
def get_Image():

    #...
    #process image with cv2 then save so I can send it the the browser 
    #...
    
    #image = get_encoded_img("img\truthMask.png")
    image = get_encoded_img("upload/prediction/sarah_connor_grayscale.jpg_out.PNG")  # I use Linux path with `/` instead of `\`
    
    return jsonify({'image_url': image})

def get_encoded_img(image_path):
    img = Image.open(image_path, mode='r')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return my_encoded_img


if __name__ == "__main__":
    app.run(debug=True)
