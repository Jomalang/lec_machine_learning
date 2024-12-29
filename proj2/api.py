from typing import Union
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# 사진을 주고받는 인터페이스를 만들어보자.
from typing import Annotated
import numpy as np
import cv2
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile

# # fast api에서 파일을 주고 받는 방법은 크게 두 가지가 있다.

# # 동기적 방식(File())
# # 이 방식은 요청에서 헤더를 받고, 이미지 바이너리 파일을 받는걸 
# # 순차적으로 기다린다.
# @app.post("/files/")
# async def create_file(file: Annotated[bytes, File()]):
#     return {"file_size": len(file)}

app = FastAPI()
# 비동기적 방식(UploadFile 객체)
# 이 방식은 요청에서 헤더를 받고, 파일을 받는걸 비동기적으로 처리한다.
# 따라서 다수의 파일을 동시에 받을 수 있음.
# 또한 헤더를 미리 읽고 원하지 않는 타입의 파일이면 예외처리도 가능
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    # 파일 바이너리 파일이 모두 도착할때까지 대기했다가 contents에 할당
    # 현재 메모리상에 이미지 바이너리 존재
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    # 바이너리 파일이 컬러인지 흑백인지 명시
    cv_mat = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 아래의 사용법이 mediapipe의 이미지 읽어오는 정석적인 방법이다. openCV를 이용해야 한다.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)

    return {"filename": result}