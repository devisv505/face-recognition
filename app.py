from threading import Thread

from flask import Flask, render_template, Response, request

from camera import Video
from face_recognizer import Recognizer
from user_pics import UserPics
from users import Users

app = Flask(__name__, static_url_path='/images')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/recognize', methods=['GET'])
def recognize():
    return render_template('recognizer.html')


@app.route('/load', methods=['GET'])
def load_faces_index():
    return render_template('load.html')


# a custom function that blocks for a moment
def upload():
    # block for a moment
    recognizer = Recognizer()
    recognizer.upload()
    # display a message
    print('This is from another thread')


@app.route('/load/execute', methods=['POST'])
def load_faces():
    thread = Thread(target=upload)
    thread.start()

    return Response("OK")


@app.route('/users', methods=['GET'])
def all_users():
    users = Users()
    return render_template('users.html', users=users.get_all())


def replace(pic):
    return pic.replace("./static/", "/")


@app.route('/users/pics', methods=['GET'])
def user_pics():
    pics = UserPics()
    pics_url = map(replace, pics.get_all_pics(request.args["name"]))

    print(request.args["name"])
    print(pics_url)
    return render_template('user_pics.html', pics=pics_url)


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n'
               )


@app.route('/video')
def video():
    return Response(gen(Video()), mimetype='multipart/x-mixed-replace; boundary=frame')


app.run(debug=True)
