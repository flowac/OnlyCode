import os, sys, time, uuid
from flask import Flask, jsonify, request
import count

upload_dir = "tmp/"
valid_ext = {"jpg", "png"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = upload_dir
app.config["MAX_CONTENT_LENGTH"] = 1 * 1000 * 1000
port = 7003
os.makedirs(upload_dir, exist_ok=True)
wc = count.count()

def file_ok(fname):
    ext = fname.rsplit('.', 1)[-1].lower()
    return ext in valid_ext

@app.route("/", methods=["GET", "POST"])
def proc_image():
    value = None
    if request.method == "GET":
        return "ready", 200

    try:
        file = request.files["file"]
        if not file_ok(file.filename):
            return "invalid", 204
        fname = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))

        file.save(fname)
        value = wc.run(fname)
        os.remove(fname)
    except OSError as e:
        print(f"Error: An unexpected error occurred: {e}")

    if value is None:
        return "invalid", 204
    return jsonify(value=value), 200

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=port)

