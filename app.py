import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
import base64
import io

app = Flask(__name__)

IMG_SIZE = (227,227)

# Load model
model = tf.keras.models.load_model("32 -2048-1024/alexnet_flower_final.keras", compile=False)
class_names = np.load("class_names.npy", allow_pickle=True)


def predict_image(img):

    img = img.resize(IMG_SIZE)
    img_array = np.array(img)/255.0
    img_batch = np.expand_dims(img_array,axis=0)

    preds = model.predict(img_batch)[0]

    # lấy top 5
    top5_idx = preds.argsort()[-5:][::-1]

    results = []

    for i in top5_idx:

        results.append({
            "class": class_names[i],
            "prob": float(preds[i]*100)
        })

    return results


@app.route("/",methods=["GET","POST"])
def index():

    label=None
    conf=None
    results=None
    img_data=None

    if request.method=="POST":

        file=request.files["file"]

        if file:

            img = Image.open(file.stream).convert("RGB")

            # predict
            results = predict_image(img)

            label = results[0]["class"]
            conf = results[0]["prob"]

            # convert image -> base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_data = base64.b64encode(buffer.getvalue()).decode()

    return render_template(
        "index.html",
        label=label,
        conf=conf,
        results=results,
        img_data=img_data
    )


if __name__=="__main__":
    app.run(debug=True)