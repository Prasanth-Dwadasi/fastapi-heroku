from datetime import datetime
from fastapi import FastAPI,File, UploadFile
from fastapi.responses import FileResponse,Response
import os
import zipfile
import io
from io import BytesIO
from typing import Optional,List
from pydantic import BaseModel
from assignment3 import predict
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/")
def read_main():
    return {"message":"Pass the location to /get_predictions_json to get output"}

class Inputs(BaseModel):
    location: str
    starttime: Optional[datetime] = None

# @app.post("/get_predictions_json/")
# def get_predictions_json(input:Inputs):
#     input_dict = input.dict()
#     location = input_dict["location"]
#     file = predict(location)
#     if file:
#         return zipfiles(file)
#     else:
#         return {"Error":"Location not available"}

@app.post("/get_predictions_json/")
def get_predictions_json(input:Inputs):
    input_dict = input.dict()
    location = input_dict["location"]
    file = predict(location)
    if file:
        with open(file, 'rb') as f:
            img_raw = f.read()
        byte_io = BytesIO(img_raw)
        return StreamingResponse(byte_io, media_type='image/gif')
    else:
        return {"Error":"Location not available"}

# @app.post("/get_predictions/{location}")
# def get_predictions(location):
#     file = predict(location)
#     return zipfiles(file)


@app.get("/get_predictions/{location}")
def get_predictions(location):
    file = predict(location)
    with open(file, 'rb') as f:
        img_raw = f.read()
    byte_io = BytesIO(img_raw)
    return StreamingResponse(byte_io, media_type='image/gif')


# def zipfiles(file_path):
#     zip_filename = "archive.zip"

#     s = io.BytesIO()
#     zf = zipfile.ZipFile(s, "w")

#     # for fpath in filenames:
#     #     # Calculate path for file in zip
#     #     fdir, fname = os.path.split(fpath)

#     #     # Add file, at correct path
#     #     zf.write(fpath, fname)

#     for file in os.listdir(file_path):
#         if file.endswith(".jpg"):
#             zf.write(file_path +"/" + file)

#     # Must close zip for all contents to be written
#     zf.close()

#     # Grab ZIP file from in-memory, make response with correct MIME-type
#     resp = Response(s.getvalue(), media_type="application/x-zip-compressed", headers={
#         'Content-Disposition': f'attachment;filename={zip_filename}'
#     })

#     return resp




# @app.post("/upload")
# async def upload(files: List[UploadFile] = File(...)):

#     # in case you need the files saved, once they are uploaded
#     for file in files:
#         contents = await file.read()
#         save_file(file.filename, contents)

#     return {"Uploaded Filenames": [file.filename for file in files]}

# def save_file(filename, data):
#     with open(filename, 'wb') as f:
#         f.write(data)


# @app.get("/download_images")
# def image_endpoint():
#     images = ["pic.jpg","Picture1.png"]
#     return zipfiles(images)





