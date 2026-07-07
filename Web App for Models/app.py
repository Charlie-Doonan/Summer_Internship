import os
import cv2
import numpy as np
import base64
import subprocess
from flask import Flask, request, jsonify, render_template
from segment import find_stem

app = Flask(__name__)

BASE_DIR = r"C:\Users\charl\Desktop\Models"

PYTHON_ENVS = {
    "sam": r"C:\Users\charl\anaconda3\envs\segment-anything\python.exe",
    "sam3": r"C:\Users\charl\anaconda3\envs\sam3\python.exe",
    "vggt": r"C:\Users\charl\anaconda3\envs\vggt\python.exe"
}

MODEL_SCRIPTS = {
    "sam": r"C:\Users\charl\Desktop\Models\segment-anything\scripts\segment_stem.py",
    "sam3": r"C:\Users\charl\Desktop\Models\sam3\segment_stem.py",
    "vggt": r"C:\Users\charl\Desktop\Models\vggt-omega\segment_stem.py"
}


def encode(img):
    if img is None:
        return ""

    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():

    file = request.files["image"]
    model = request.form.get("model")

    valid_models = ["segment", "sam", "sam3", "vggt"]

    if model not in valid_models:
        return jsonify({"error": "invalid model"}), 400

    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        return jsonify({"error": "invalid image"}), 400


    if model == "segment":

        result, mask = find_stem(img)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        return jsonify({
            "result": encode(result),
            "mask": encode(mask_bgr)
        })


    input_path = os.path.abspath("temp_input.jpg")
    output_path = os.path.abspath("temp_output.jpg")

    cv2.imwrite(input_path, img)

    script_path = MODEL_SCRIPTS[model]
    python_path = PYTHON_ENVS[model]
    model_dir = os.path.dirname(script_path)

    print("\n--- MODEL RUN START ---")
    print("Model:", model)
    print("Python:", python_path)
    print("Script:", script_path)
    print("Working dir:", model_dir)
    print("----------------------\n")

    try:

        subprocess.run(
            [
                python_path,
                script_path,
                "--input",
                input_path,
                "--output",
                output_path
            ],
            cwd=model_dir,
            check=True
        )

    except subprocess.CalledProcessError as e:

        print("MODEL FAILED:", str(e))

        return jsonify({
            "error": "model failed",
            "details": str(e)
        }), 500

    print("MODEL FINISHED\n")

    result = cv2.imread(output_path)

    if result is None:
        return jsonify({
            "error": "no output image generated"
        }), 500

    return jsonify({
        "result": encode(result)
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
