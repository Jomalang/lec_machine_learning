
# STEP 1 : 필수 모듈 임포트
import easyocr

# STEP 2 : 추론기 객체 생성
reader = easyocr.Reader(['ko','en']) # this needs to run only once to load the model into memory

# STEP 3 : 데이터 로드
data = 'ganpan.jpg'

# STEP 4 : 추론 실행
# detail 파라미터는 반환할 디테일의 수준을 결정한다.
result = reader.readtext(data, detail=0)
print(result)

# STEP 5 : 결과 후처리
# if ... : 
#     ...