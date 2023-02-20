from flask import Flask, request, send_file, send_from_directory
import os
import random
from PIL import Image, ImageEnhance, ImageDraw
from flask_cors import CORS
import time
import tempfile
import shutil
import cv2

app = Flask(__name__)
cors = CORS(app)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["IMAGE_FOLDER"] = "images"
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

@app.route("/generate", methods=["POST"])
def generate_wallpaper():
    start_time = time.time()
    # Load the user's uploaded image
    image_file = request.files["image"]
    names = request.form.getlist("names")

    # Save the uploaded image
    image_file.save(os.path.join(app.config["UPLOAD_FOLDER"], "upload.png"))

    # Load the face images and select the ones with matching names
    face_images = [
    ]

    for name in names:
        image_path = os.path.join(app.config["IMAGE_FOLDER"], f"{name}.jpg")
        if os.path.isfile(image_path):
            face_image = Image.open(image_path)
        else:
            continue
        face_images.append(face_image)

    resize_and_save_image(os.path.join(
        app.config["UPLOAD_FOLDER"], "upload.png"))

    # Load the original image
    original_image = Image.open(os.path.join(
        app.config["UPLOAD_FOLDER"], "upload.png"))

    # Convert the mode of the original image to 'RGBA' if it does not have an alpha channel
    if original_image.mode != 'RGBA':
        original_image = original_image.convert('RGBA')

    # Paste the face images on the original image
    num_copies = random.randint(1, 5)
    for i in range(num_copies):
        for i, face_image in enumerate(random.sample(face_images, min(16, len(face_images)))):
            # Select a random position and rotation for the face image
            angle = random.randint(-30, 30)
            if face_image.width > original_image.width or face_image.height > original_image.height:
                # Resize the face image to fit within the original image
                face_image = face_image.resize(
                    (original_image.width // 2, original_image.height // 2))

            # Rotate the face image
            rotated_image = face_image.rotate(
                angle, expand=True, fillcolor=(0, 0, 0, 0))
            optimal_size = calculate_optimal_size(rotated_image.size, 175)
            rotated_image = rotated_image.resize(optimal_size)

            # Remove the black background
            mask = Image.new("L", rotated_image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([(0, 0), rotated_image.size],
                           fill=255, outline=None)
            for x in range(rotated_image.width):
                for y in range(rotated_image.height):
                    if rotated_image.getpixel((x, y)) == (0, 0, 0):
                        draw.point((x, y), fill=0)

            # Paste the rotated image on the original image
            rotated_image.putalpha(mask)
            position = (
                random.randint(0, original_image.width - rotated_image.width),
                random.randint(0, original_image.height - rotated_image.height)
            )
            original_image.alpha_composite(rotated_image, position)

    # Enhance the colors and brightness of the wallpaper
    enhancer = ImageEnhance.Color(original_image)
    original_image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Brightness(original_image)
    original_image = enhancer.enhance(1.2)

    wallpaper_path = os.path.join(app.config["UPLOAD_FOLDER"], "wallpaper.png")
    original_image.save(wallpaper_path)

    # Return the URL of the wallpaper file
    url = f"http://127.0.0.1:5000/{wallpaper_path}"
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time, "seconds")
    return url


def calculate_optimal_size(original_size, max_size):
    width, height = original_size
    if width <= max_size and height <= max_size:
        # The image is already smaller than the maximum size, no need to resize it
        return original_size
    if width > height:
        # Image is wider than it is tall, so we should resize its width to the maximum size and its height proportionally
        return (max_size, int(height / width * max_size))
    else:
        # Image is taller than it is wide, so we should resize its height to the maximum size and its width proportionally
        return (int(width / height * max_size), max_size)


def resize_image(img, max_width=1920, max_height=1080):
    width, height = img.size
    if width > max_width or height > max_height:
        aspect_ratio = width / height
        if aspect_ratio > max_width / max_height:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        img = img.resize((new_width, new_height))
    return img


def save_image(img, path):
    img.save(path)


def resize_and_save_image(image_path):
    with Image.open(image_path) as img:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img = resize_image(img, max_width=1920, max_height=1080)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
            temp_path = temp.name
            save_image(img, temp_path)
        upload_dir = os.path.dirname(image_path)
        upload_path = os.path.join(upload_dir, os.path.basename(image_path))
        shutil.copy2(temp_path, upload_path)
    os.remove(temp_path)

def calculate_face_center(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale for faster processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Detect faces in the image using the face detection model
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        # If no faces are detected, return None
        return None

    # Calculate the center of the first detected face
    (x, y, w, h) = faces[0]
    center_x = x + w/2
    center_y = y + h/2

    # Return the center coordinates as a tuple
    return (center_x, center_y)

UPLOAD_FOLDER = os.path.abspath("uploads")


@app.route("/uploads/<path:path>")
def serve_upload(path):
    return send_from_directory(UPLOAD_FOLDER, path)
