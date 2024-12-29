# STEP 1 : 핵심 모듈 임포트
from insightface.app import FaceAnalysis # face와 관련된 tasks를 처리하는 추론 패키지(모델 포함)
from insightface.data import get_image as ins_get_image # 테스트 데이터
from typing import Annotated
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile

# STEP 2 : 추론기 객체 생성(옵션 포함)
face = FaceAnalysis()
face.prepare(ctx_id=0, det_size=(640,640))

app = FastAPI()

# 비동기적 방식(UploadFile 객체)
@app.post("/uploadfile/")
async def create_upload_file(file1: UploadFile, file2: UploadFile):

    # 파일 바이너리 파일이 모두 도착할때까지 대기했다가 contents에 할당
    # 현재 메모리상에 이미지 바이너리 존재
    contents1 = await file1.read()
    contents2 = await file2.read()
    nparr1 = np.fromstring(contents1, np.uint8)
    nparr2 = np.fromstring(contents2, np.uint8)

    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    # STEP 4 : 추론 실행(get메서드)
    # 주어진 얼굴 이미지에 대한 정보를 기본 다섯 가지 모델을 통해 추론하고 결과를 반환한다.
    faces1 = face.get(img1)
    faces2 = face.get(img2)

    # validation체크, 현재 테스트 이미지는 사진당 얼굴이 하나씩 있어야 한다.
    assert len(faces1)==1
    assert len(faces2)==1

    # STEP 5 : 반환값 후처리

    # 데이터 정규화(normalized)후 임베딩 (-1 ~ 1)
    face_feat1 = faces1[0].normed_embedding
    face_feat2 = faces2[0].normed_embedding

    # 넘파이 자료구조로 래핑
    face_feat1 = np.array(np.array(face_feat1, dtype=np.float32))
    face_feat2 = np.array(np.array(face_feat2, dtype=np.float32))

    # 여기서 유사도 측정(similarity measure)은  
    # cosine similirity를 이용한다. 이는 내적(dotproduct)을 통해 가능.
    sims = np.dot(face_feat1, face_feat2.T) # T는 전치행렬을 구하는 numpy 지원 매직메서드

    if sims > 0.4:
        return {"result" : "동일인입니다"}
    else:
        return {"result" : "동일인이 아닙니다"}