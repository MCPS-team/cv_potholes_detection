from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
from interface import PotholeEvent
from job_queue import JobQueue
import json
import glob
from utility import fakes_pothole_obj
import os

app = Flask(__name__)
job_queue = JobQueue(verbose=True)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/test/<hash>', methods=['GET'])
def test(hash):
    for i in range(5):
        job_queue.add(hash)
    print("End test main")
    return jsonify(hash=hash, success=True)


@app.route('/analyze', methods=['POST'])
@app.route('/analyze/<path:json_filename>', methods=['GET'])
def analyze(json_filename=None):
    # if GET and File
    if json_filename is not None:
        with open(json_filename, 'r') as f:
            content = f.read()
            parsed_json = json.loads(content)
    else:
        parsed_json = request.json
    # for x in parsed_json:
    #     print("{}: {}".format(x, parsed_json[x]))
    pothole_events = [PotholeEvent().from_dict(x) for x in parsed_json]
    job = job_queue.add(pothole_events)
    return jsonify(success=True, job_id=str(job.uuid))

@app.route('/analyze_folder/<path:path>', methods=['GET'])
def analyze_folder(path):
    frames = glob.glob(os.path.join(path, '*.jpg'))
    parsed_json = [fakes_pothole_obj(frames)]
    # for x in parsed_json:
    #     print("{}: {}".format(x, parsed_json[x]))
    pothole_events = [PotholeEvent().from_dict(x) for x in parsed_json]
    job = job_queue.add(pothole_events)
    return jsonify(success=True, job_id=str(job.uuid))


@app.errorhandler(Exception)
def internal_error(e):
    req = request.get_json()
    if isinstance(e, HTTPException):
        return jsonify({
            "succes": False,
            "error": {
                "code": e.code,
                "name": e.name,
                "description": e.description,
            }
        }), HTTPException.code
    else:
        return jsonify({
            "succes": False,
            "error": {
                "description": str(e),
            }
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
