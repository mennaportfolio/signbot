from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from ultralytics import YOLO


app = FastAPI()
model = YOLO("best (7).pt")

# Define a function to resize and process the image
def process_image(image_path) -> str:
    # For example, you could use a pre-trained model to classify the image
    results = model(image_path)
    for result in results:
        if result.boxes:
            box = result.boxes[0]
            class_id = int(box.cls)
            object_name = model.names[class_id]
    print(object_name)
    # for r in result:
    #     print(r.probs)
    # print(result)
    return object_name

@app.post("/upload/")
async def upload_file(reference_image: UploadFile = File(...)):
    try:
        # Read the uploaded file directly into a PIL Image
        image = Image.open(BytesIO(await reference_image.read()))
        
        # Resize the image to 640x640
        # resized_image = image.resize((1024, 1024))
        
        # Save the resized image to the local directory
        save_path = "uploaded_image.jpg"
        image.save(save_path)
        
        reference_processed_img = process_image(image)        
        
        # Return the processed result
        return {"result": reference_processed_img}
    except Exception as e:
        # If an error occurs, return an error message
        return JSONResponse(status_code=500, content={"error": str(e)})
