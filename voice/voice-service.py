import os, sys, time
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
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
    return fname.rsplit('.', 1)[-1].lower() in valid_ext

@app.route("/", methods=["GET", "POST"])
def proc_voice():
    if request.method == "GET":
        return "ready", 200

    file = request.files["file"]
    fname = secure_filename(file.filename)
    if not file_ok(fname):
        return "invalid", 204

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
    density, impact = vc.run(fname)
    try:
        os.remove(upload_dir + fname)
        print(f"Deleted '{upload_dir}{fname}'.")
    except FileNotFoundError:
        print(f"Error: File '{upload_dir}{fname}' not found.")
    except PermissionError:
        print(f"Error: Permission denied to delete '{upload_dir}{fname}'.")
    except OSError as e:
        print(f"Error: An unexpected error occurred: {e}")

    if density is None or impact is None:
        return "invalid", 204
    return jsonify(density=density, impact=impact), 200

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=port)

# status check: curl -X GET http://127.0.0.1:7001
# send request: curl -X POST -F "file=@terry.mp3" 127.0.0.1:7001
# upload limit: curl -X POST -F "file=@micro-machine.wav" 127.0.0.1:7001
# format check: curl -X POST -F "file=@badname" 127.0.0.1:7001
# corrupted   : curl -X POST -F "file=@bad.wav" 127.0.0.1:7001

