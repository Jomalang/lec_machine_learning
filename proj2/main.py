# face recognition 태스크를 평정한 오픈소스 모델이 있다. 바로 insight face임... 그런데 이 모델은 설치가 어렵기로 악명이 높다.

# STEP 1 : 핵심 모듈 임포트
import cv2
import numpy as np
from insightface.app import FaceAnalysis # face와 관련된 tasks를 처리하는 추론 패키지(모델 포함)
from insightface.data import get_image as ins_get_image # 테스트 데이터

# 버전 체크하기 위한 코드
# assert insightface.__version__>='0.3'

# 아래의 코드는 .py파일를 터미널에서 읽힐때 같이 제공한 args를 읽는 부분임. 지금은 사용하지 않음
# parser = argparse.ArgumentParser(description='insightface app test')
# ctx는 gpu사이즈
# parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
# det-size는 얼굴찾을때의 사이즈?
# parser.add_argument('--det-size', default=640, type=int, help='detection size')
# args = parser.parse_args()

# 이하부터 추론에 쓰이는 핵심 코드
# STEP 2 : 추론기 객체 생성(옵션 포함)
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))


# STEP 3 : 입력 데이터 로드
# img = ins_get_image('t1')
img1 = cv2.imread("han.jpg")
img2 = cv2.imread("kim.jpg")


# STEP 4 : 추론 실행(get메서드)
# 주어진 얼굴 이미지에 대한 정보를 기본 다섯 가지 모델을 통해 추론하고 결과를 반환한다.
# faces = app.get(img)
faces1 = app.get(img1)
faces2 = app.get(img2)

# validation체크, 현재 테스트 이미지는 사진당 얼굴이 하나씩 있어야 한다.
assert len(faces1)==1
assert len(faces2)==1

# STEP 5 : 반환값 후처리
# STEP 5-1 : 결과값 저장(기존 visualization util사용하는 것과 동일)
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)

# then print all-to-all face similarity
# STEP 5-2 :얼굴간의 상호유사도 비교
# feats = []
# for face in faces:
    # feats.append(face.normed_embedding)
# feats = np.array(feats, dtype=np.float32)

# 데이터 정규화(normalized)후 임베딩 (-1 ~ 1)
face_feat1 = faces1[0].normed_embedding
face_feat2 = faces2[0].normed_embedding

# 넘파이 자료구조로 래핑
face_feat1 = np.array(np.array(face_feat1, dtype=np.float32))
face_feat2 = np.array(np.array(face_feat2, dtype=np.float32))

# 여기서 유사도 측정(similarity measure)은  
# cosine similirity를 이용한다. 이는 내적(dotproduct)을 통해 가능.
sims = np.dot(face_feat1, face_feat2.T) # T는 전치행렬을 구하는 numpy 지원 매직메서드
print(sims)