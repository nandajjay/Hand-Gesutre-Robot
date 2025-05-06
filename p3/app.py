# app.py
from flask import Flask, render_template, Response, jsonify
from gesture_control import GestureController
import threading

app = Flask(__name__)
controller = GestureController()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    while True:
        frame = controller.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/status')
def get_status():
    return jsonify(controller.get_status())

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        controller.stop()