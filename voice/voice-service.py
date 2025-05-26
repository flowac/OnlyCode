import os, sys, time, uuid
from flask import Flask, jsonify, request
import voice

upload_dir = "tmp/"
valid_ext = {"mp3", "wav"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = upload_dir
app.config["MAX_CONTENT_LENGTH"] = 2 * 1000 * 1000
port = 7001
os.makedirs(upload_dir, exist_ok=True)
vc = voice.voice()

def file_ok(fname):
    ext = fname.rsplit('.', 1)[-1].lower()
    return ext in valid_ext

@app.route("/", methods=["GET", "POST"])
def proc_voice():
    density = None
    impact = None
    if request.method == "GET":
        return "ready", 200

    try:
        file = request.files["file"]
        if not file_ok(file.filename):
            return "invalid", 204
        fname = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))

        file.save(fname)
        density, impact = vc.run(fname)
        os.remove(fname)
    except OSError as e:
        print(f"Error: An unexpected error occurred: {e}")

    if density is None or impact is None:
        return "invalid", 204
    return jsonify(density=density, impact=impact), 200

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=port)

