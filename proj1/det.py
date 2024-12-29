#@markdown We implemented some functions to visualize the object detection results. <br/> Run the following cell to activate the functions.
import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


# 화면에 뿌리기 위한 함수
def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

# img = cv2.imread('cat_and_dog.jpg')
# # cv2_imshow(img)

# STEP 1: 필수 모듈 임포트
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: 옵션을 포함해 추론기 객체 생성
base_options = python.BaseOptions(model_asset_path='models\efficientdet_lite0.tflite')
# 객체를 찾을때의 문턱값 설정
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# STEP 3: 인풋 이미지 입력
image = mp.Image.create_from_file('cat_and_dog.jpg')

# STEP 4: 객체 찾기 추론 실행
detection_result = detector.detect(image)

# STEP 5: 결과 후 처리
# 이미지 복사
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
# OpenCV는 이미지 처리할때 RGB가 아니라 BGR로 거꾸로 처리된다.
# 그래서 화면에 출력하려면 이를 한번 뒤집어줘야함.
# 그래서 convertColor메서드가 필요하다.
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
cv2.imshow('test',rgb_annotated_image)
cv2.waitKey(0)
print(detection_result)