from flask import Flask, request, render_template, Response
from camera import VideoCamera, predict_emotion

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    

@app.route('/emotion_pred', methods=['POST'])
def emotion_pred():
    data = request.get_json()

    return predict_emotion(data['image'])

if __name__ == '__main__':
    app.run(debug=True)
