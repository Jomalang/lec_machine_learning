# STEP 1 : 모듈 임포트
from transformers import pipeline
from typing import Annotated
from fastapi import FastAPI, Form

# STEP 2 : 추론기 객체 생성
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")

app = FastAPI()

@app.post("/inference/")
async def inference(text : Annotated[str, Form()]):
    # SETP 3 : 입력 데이터 로드(파라미터에 있음.)

    # STEP 4 : 추론 실행
    result = classifier(text)

    # STEP 5 : 결과 후처리
    return {"username": result}
