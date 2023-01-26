from fastapi import FastAPI,File,UploadFile
import uvicorn
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
app = FastAPI()

model=tf.keras.models.load_model("../pythonProject2/1")

class_name=["Early Blight","Late Blight","Healthy"]



def read_file_as_image(data) -> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
        file: UploadFile=File(...)
):
    image =read_file_as_image(await file.read())
    #it take as batch to to make it like batch to expand it
    img_batch=np.expand_dims(image,0)
    prediction=model.predict(img_batch)
    pre_class=class_name[np.argmax(prediction[0])] #zero e take bcz prediction is 2 d array

    #now for confidence
    confedense=np.max(prediction[0])

    return {
        'class':pre_class,
        'confidence':float(confedense)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
