
# STEP 1 : 필수 모듈 임포트
import easyocr
import requests
from bs4 import BeautifulSoup
import re

# STEP 2 : 추론기 객체 생성
reader = easyocr.Reader(['ko','en']) # this needs to run only once to load the model into memory

# STEP 3 : 데이터 로드
data = 'iliad.jpg'

# STEP 4 : 추론 실행
# detail 파라미터는 반환할 디테일의 수준을 결정한다.
result = reader.readtext(data, detail=0)
print(result)

# STEP 5 : 결과 후처리

# 영어 제목과 한글 제목중 영어 제목을 더 우선시 합니다.
# 영어 알파벳 검증 함수
def is_english_alphabet(s):
    return bool(re.fullmatch(r"[A-Za-z]+", s))

# 한글 알파벳 검증 함수
def is_korean_alphabet(s):
    return bool(re.fullmatch(r"[가-힣]+", s))

# 영어 제목 추출
for keyword in result:
    flag = True
    for word in keyword:
        if not is_english_alphabet(word):
            flag = False
            break
    if flag:
        book_title = keyword
        break

# 한글 제목 추출
if not book_title:
    for keyword in result:
        flag = True
        for word in keyword:
            if not is_korean_alphabet(word):
                flag = False
                break
        if flag:
            book_title = keyword
            break

print(f"검색할 책 제목: {book_title}")
            
# 책 제목과 API 키 설정
API_KEY = "AIzaSyBRIqgaYx6gND88TgipSBhp2Dp4EDgYbuc"

# 요청 URL
url = f"https://www.googleapis.com/books/v1/volumes?q={book_title}&key={API_KEY}"

# API 호출
response = requests.get(url)
data = response.json()

# 결과 출력
if 'items' in data:
        description = data['items'][0]['volumeInfo'].get('description', '설명 없음')
        
        print(f"설명: {description}\n")
else:
    print("해당 책을 찾을 수 없습니다.")