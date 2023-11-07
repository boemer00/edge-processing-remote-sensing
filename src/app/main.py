from fastapi import FastAPI, File, UploadFile, HTTPException
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = FastAPI()

# Load the model
model = load_model('best_model.h5')

# Read and preprocess an image
def read_image_file(file) -> np.ndarray:
    try:
        img = image.load_img(file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

# Define a dictionary to convert numerical predictions to labels
labels = {0: 'desert', 1: 'water', 2: 'green_area', 3: 'cloudy'}

@app.post("/predict/")
async def create_upload_file(file: UploadFile = File(...)):
    # Ensure the file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not an image.")

    try:
        # Read the image file using the utility function
        image_data = await read_image_file(await file.read())

        # Make prediction
        predictions = model.predict(image_data)

        # Convert one-hot encoded prediction to label
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_label = labels[predicted_index]

        return {"filename": file.filename, "prediction": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Edge Prediction API!"}
