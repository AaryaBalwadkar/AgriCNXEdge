from io import BytesIO
import numpy as np
from PIL import Image, ImageOps
from app.core.model_metadata import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD


def preprocess_image(content: bytes) -> np.ndarray:
    image = Image.open(BytesIO(content))
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(image).astype("float32") / 255.0
    array = (array - np.array(IMAGE_MEAN, dtype="float32")) / np.array(IMAGE_STD, dtype="float32")
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0).astype("float32")
