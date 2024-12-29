# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision
from typing import Union
from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2

# STEP 2: Create an ImageClassifier object.

# 사용할 모델 경로 입력
base_options = python.BaseOptions(model_asset_path='models\\efficientnet_lite0.tflite')

# max_results = 상위 몇개의 결과만 반환할 것인지
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=4) 

# 옵션을 이용해 객체 생성
classifier = vision.ImageClassifier.create_from_options(options)


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
    # STEP 3: Load the input image.
    # 읽어올 이미지 입력(OpenCV로 읽어온다.)
    # cv2의 imread()는 file_open + image_decode 함수가 합쳐져있다. 
    # 따라서 지금 상황에서는 decode만 해줘도 된다.(바이너리파일이 메모리에 적재되어있음.)
    # cv_mat = cv2.imread(input_file) << 즉, 이걸 안써도 된다.
    # 그런데 웹 요청으로 온 이미지는 Text이므로, 바이너리로 바꿔줘야 한다.
    # 이때는 numpy까지 필요하다... 
    nparr = np.fromstring(contents, np.uint8)
    # 바이너리 파일이 컬러인지 흑백인지 명시
    cv_mat = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 아래의 사용법이 mediapipe의 이미지 읽어오는 정석적인 방법이다. openCV를 이용해야 한다.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)

    # STEP 4: Classify the input image.
    # 생성된 추론기 객체의 classify메서드 사용
    # 우리가 가져온 모델이 classifier이기 때문에 메서드명이 classify인것. tasks별로 메서드 명이 다르다.
    classification_result = classifier.classify(image)
    # print(classification_result)

    # STEP 5: Process the classification result. In this case, visualize it.
    # 반환된 결과를 후처리 하는 부분
    top_category = classification_result.classifications[0].categories[0]
    result = f"{top_category.category_name} ({top_category.score:.2f})"

    return {"filename": result}