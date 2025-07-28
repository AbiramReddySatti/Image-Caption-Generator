from flask import Flask, render_template, request
from model import generate_caption
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    if request.method == 'POST':
        image = request.files['image']
        if image:
            image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(image_path)
            caption = generate_caption(image_path)
    return render_template('index.html', caption=caption)

if __name__ == '__main__':
    app.run(debug=True) 