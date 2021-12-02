from flask import Flask, request, render_template

# this is the python backend
# create the web app
app = Flask(__name__)


# routes are the endpoints
# they are the urls associated with the website
# url name is in the parentheses
# the function is called when the url is loaded
@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/predictimage/', methods=["GET", "POST"])
def predict_image():
    return render_template('prediction.html')


app.run(debug=True)
