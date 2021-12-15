import os
import base64
import io
from flask import Flask, request, flash, render_template, redirect, jsonify
from werkzeug.utils import secure_filename
from prediction import doPrediction
from PIL import Image

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/examples/')
def examples():
    return render_template('examples.html')

@app.route('/info/')
def info():
    return render_template('info_page.html')

@app.route('/authors/')
def authors():
	return render_template('authors.html')

@app.route('/video/')
def video():
	return render_template('video.html')
	
@app.route('/poster/')
def poster():
	return render_template('poster.html')
	
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
            modelType = request.form['model']
            img = doPrediction(filename, modelType)
            image = get_encoded_img(img)  # I use Linux path with `/` instead of `\`
    
            return jsonify({'image_url': image})

    return render_template('upload.html')
	
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_encoded_img(image_path):
    img = Image.open(image_path, mode='r')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return my_encoded_img
	

if __name__ == "__main__":
    app.run(debug=True)
