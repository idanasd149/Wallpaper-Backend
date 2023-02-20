from flask import Flask, request, send_file, send_from_directory
import os
import random
from PIL import Image, ImageEnhance, ImageDraw
from flask_cors import CORS
import time
import tempfile
import shutil
import cv2
import rembg
import io

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
    face_model = request.form.get("face_model_on")

    # Save the uploaded image
    image_file.save(os.path.join(app.config["UPLOAD_FOLDER"], "upload.png"))

    face_images = [
    ]

    # Appending all the names to array of images
    for name in names:
        image_path = os.path.join(app.config["IMAGE_FOLDER"], f"{name}.png")
        if os.path.isfile(image_path):
            face_image = Image.open(image_path)
        else:
            continue
        face_images.append(face_image)

    face_images = process_image(face_images, 1)
    # Path for wallpaper
    wallpaper_path = os.path.join(app.config["UPLOAD_FOLDER"], "wallpaper.png")

    # Resizing the original image
    resize_and_save_image(os.path.join(
        app.config["UPLOAD_FOLDER"], "upload.png"))

    # Load the original image
    original_image = Image.open(os.path.join(
        app.config["UPLOAD_FOLDER"], "upload.png"))

    # Call the calculate_face_center function to detect faces in the image
    face_centers = calculate_face_center(os.path.join(
        app.config["UPLOAD_FOLDER"], "upload.png"))

    if face_model == 'true':
        print('face found')
        # If a face is detected, paste a random face image on the original image
        original_image = paste_face_image(
            original_image, face_images, face_centers)

    else:
        print('face not found')
        # If no face is detected, paste a random face image on the original image at a random position and rotation
        original_image = paste_random_image(original_image, face_images)

    # Save the modified image to a file
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


def paste_face_image(original_image, face_images, face_centers):
    # Create a new Image object with the same size and mode as the original image
    canvas = Image.new(mode=original_image.mode, size=original_image.size)

    # Paste the original image onto the new Image object
    canvas.paste(original_image, (0, 0))

    # Iterate over all face centers
    for face_center in face_centers:
        # Select a random face image
        face_image = random.sample(face_images, min(16, len(face_images)))[0]
        # Rotate the face image and resize
        angle = random.randint(-30, 30)
        rotated_image = rotate_face_image(face_image, angle)
        # Resize the face image to fit within the original image
        max_width = original_image.width - face_center[0]
        max_height = original_image.height - face_center[1]
        rotated_image = resize_face_image(rotated_image, max_width, max_height)

        # Calculate the position to paste the face image
        position = (
            int(face_center[0] - rotated_image.width / 2),
            int(face_center[1] - rotated_image.height / 2)
        )

        # Paste the rotated image onto the new Image object
        canvas.alpha_composite(rotated_image, position)
    return canvas


def paste_random_image(original_image, face_images):
    num_copies = random.randint(1, 5)
    for face_image in face_images:
        for i in range(num_copies):
            angle = random.randint(-30, 30)
            rotated_image = rotate_face_image(face_image, angle)

            # Remove the black background and paste the rotated image at a random position on the original image
            position = get_random_position(original_image, rotated_image)
            original_image.alpha_composite(rotated_image, position)
    return original_image


def get_paste_position(center, face_image):
    return (
        int(center[0] - face_image.width / 2),
        int(center[1] - face_image.height / 2)
    )


def calculate_face_center(image_path, probability_threshold=0.5):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier("holyshitfacedetectionxml.xml")

    # Convert the image to grayscale for faster processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image using the face detection model
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        # If no faces are detected, return an empty array
        return []

    # Calculate the probability of each detected face based on the size of the face
    max_size = max([w*h for (x, y, w, h) in faces])
    probabilities = [(w*h) / max_size for (x, y, w, h) in faces]

    # Create an array to store the center coordinates of each detected face
    centers = []

    # Iterate over all detected faces
    for i in range(len(faces)):
        # Check if the probability of the current face is above the threshold
        if probabilities[i] >= probability_threshold:
            print(probabilities[i])
            # Calculate the center coordinates of the current face
            (x, y, w, h) = faces[i]
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)

            # Add the center coordinates to the array
            centers.append((center_x, center_y))

    # Return the array of center coordinates
    return centers


def resize_face_image(face_image, max_width, max_height):
    width, height = face_image.size
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return face_image.resize((new_width, new_height))
    else:
        return face_image


def rotate_face_image(face_image, angle):
    rotated_image = face_image.rotate(
        angle, expand=True, fillcolor=(0, 0, 0, 0))
    optimal_size = calculate_optimal_size(rotated_image.size, 300)
    rotated_image = rotated_image.resize(optimal_size)
    return rotated_image


def get_random_position(original_image, rotated_image):
    x = random.randint(0, original_image.width - rotated_image.width)
    y = random.randint(0, original_image.height - rotated_image.height)
    return (x, y)


def process_image(images, pixels_to_remove):
    modified_images = []

    for image in images:
        # Remove the background from the image using the rembg library
        img = rembg.remove(image)

        # Convert the image data to a Pillow Image object
        img = img.convert("RGBA")

        # Remove a certain number of pixels from the bottom of the image
        width, height = img.size
        img = img.crop((0, 0, width, height - pixels_to_remove))

        modified_images.append(img)

    return modified_images


UPLOAD_FOLDER = os.path.abspath("uploads")


@app.route("/uploads/<path:path>")
def serve_upload(path):
    return send_from_directory(UPLOAD_FOLDER, path)
