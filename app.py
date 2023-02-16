from flask import Flask, request, send_file, send_from_directory
import os
import random
from PIL import Image, ImageEnhance, ImageDraw
from flask_cors import CORS


app = Flask(__name__)
cors = CORS(app)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["IMAGE_FOLDER"] = "images"
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.route("/generate", methods=["POST"])
def generate_wallpaper():
    # Load the user's uploaded image
    image_file = request.files["image"]
    names = request.form.getlist("names")
    
    # Save the uploaded image
    image_file.save(os.path.join(app.config["UPLOAD_FOLDER"], "upload.png"))
    
    # Load the face images and select the ones with matching names
    face_images = [
        Image.open(os.path.join(app.config["IMAGE_FOLDER"], f"{name}.jpg"))
        for name in names if os.path.isfile(os.path.join(app.config["IMAGE_FOLDER"], f"{name}.jpg"))
    ]

    for name in names:
        image_path = os.path.join(app.config["IMAGE_FOLDER"], f"{name}.jpg")
        if os.path.isfile(image_path):
            face_image = Image.open(image_path)
            # Resize the face image to a smaller size
            face_images.append(face_image)
    
    # Load the original image
    original_image = Image.open(os.path.join(app.config["UPLOAD_FOLDER"], "upload.png"))
    
    # Convert the mode of the original image to 'RGBA' if it does not have an alpha channel
    if original_image.mode != 'RGBA':
        original_image = original_image.convert('RGBA')
    
    # Paste the face images on the original image
    for i, face_image in enumerate(random.sample(face_images, min(16, len(face_images)))):
        # Select a random position and rotation for the face image
        angle = random.randint(-30, 30)
        if face_image.width > original_image.width or face_image.height > original_image.height:
            # Resize the face image to fit within the original image
            face_image = face_image.resize((original_image.width // 2, original_image.height // 2))

        # Rotate the face image
        rotated_image = face_image.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))
        optimal_size = calculate_optimal_size(rotated_image.size, 100)
        rotated_image = rotated_image.resize(optimal_size)

        # Remove the black background
        mask = Image.new("L", rotated_image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, rotated_image.width, rotated_image.height), fill=255)
        rotated_image.putalpha(mask)

        # Paste the face image on the original image
        position = (
            random.randint(0, original_image.width - rotated_image.width),
            random.randint(0, original_image.height - rotated_image.height)
        )
        original_image.alpha_composite(rotated_image, position)

        """ # Rotate and paste the face image on the original image
        x_offset = random.randint(-original_image.width//4, original_image.width//4)
        y_offset = random.randint(-original_image.height//4, original_image.height//4)
        face_image = face_image.rotate(angle, expand=True)
        face_image = face_image.convert("RGBA")
        original_image.alpha_composite(face_image, (x_offset, y_offset))"""
    
    # Enhance the colors and brightness of the wallpaper
    enhancer = ImageEnhance.Color(original_image)
    original_image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Brightness(original_image)
    original_image = enhancer.enhance(1.2)

    wallpaper_path = os.path.join(app.config["UPLOAD_FOLDER"], "wallpaper.png")
    original_image.save(wallpaper_path)
    
    # Return the URL of the wallpaper file
    url = f"http://127.0.0.1:5000/{wallpaper_path}"
    return url

def calculate_optimal_size(original_size, max_size):
    """
    Calculates the optimal size for an image given its original size and a maximum size to resize to.

    :param original_size: Tuple of the form (width, height) representing the original size of the image.
    :param max_size: The maximum size to resize the image to, as an integer representing the width of the image in pixels.
    :return: Tuple of the form (width, height) representing the optimal size of the image after resizing.
    """
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


UPLOAD_FOLDER = os.path.abspath("uploads")

@app.route("/uploads/<path:path>")
def serve_upload(path):
    return send_from_directory(UPLOAD_FOLDER, path)