
# STEP 1 : 모듈 임포트
from transformers import pipeline

# STEP 2 : 추론기 객체 생성
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")

# SETP 3 : 입력 데이터 로드
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# STEP 4 : 추론 실행
result = classifier(text)

# STEP 5 : 데이터 후처리
print(result)
