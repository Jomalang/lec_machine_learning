# STEP 1: 필수 모듈 임포트
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: 옵션을 포함해 추론기 객체 생성
base_options = python.BaseOptions(model_asset_path='models\efficientdet_lite0.tflite')
# 객체를 찾을때의 문턱값 설정
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)


app = FastAPI()
# 비동기적 방식(UploadFile 객체)
# 이 방식은 요청에서 헤더를 받고, 파일을 받는걸 비동기적으로 처리한다.
# 따라서 다수의 파일을 동시에 받을 수 있음.
# 또한 헤더를 미리 읽고 원하지 않는 타입의 파일이면 예외처리도 가능
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    # STEP 3: 인풋 이미지 입력
    # 파일 바이너리 파일이 모두 도착할때까지 대기했다가 contents에 할당
    # 현재 메모리상에 이미지 바이너리 존재
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    # 바이너리 파일이 컬러인지 흑백인지 옵션을 주고 매트릭스 생성
    cv_mat = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 아래의 사용법이 mediapipe의 이미지 읽어오는 정석적인 방법이다. openCV를 이용해 매트릭스를 이미지로 재생성
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)

    # STEP 4: 객체 찾기 추론 실행
    detection_result = detector.detect(image)

    # STEP 5: 결과 후 처리
    total_count = len(detection_result.detections)
    person_count = 0
    for d in detection_result.detections:
        if d.categories[0].category_name == "person":
            person_count += 1
    
    result = {
        "total_count" : total_count,
        "person_count": person_count
    }

    return {"filename": result}


