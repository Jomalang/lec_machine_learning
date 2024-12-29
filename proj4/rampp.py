'''
 * The Recognize Anything Plus Model (RAM++)
 * Written by Xinyu Huang
'''

# STEP1 : 필수 모듈 임포트
import argparse
import numpy as np
import random

import torch

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform


# parser = argparse.ArgumentParser(
#     description='Tag2Text inferece for tagging and captioning')
# parser.add_argument('--image',
#                     metavar='DIR',
#                     help='path to dataset',
#                     default='images/demo/demo1.jpg')
# parser.add_argument('--pretrained',
#                     metavar='DIR',
#                     help='path to pretrained model',
#                     default='pretrained/ram_plus_swin_large_14m.pth')
# parser.add_argument('--image-size',
#                     default=384,
#                     type=int,
#                     metavar='N',
#                     help='input image size (default: 448)')




if __name__ == "__main__":

    # args = parser.parse_args()
    
    transform = get_transform(image_size=384)

    #######load model
    # STEP 2 : 추론기 객체 생성
    model_path = "ram_plus_swin_large_14m.pth"
    model = ram_plus(pretrained=model_path,
                             image_size=384,
                             vit='swin_l')
    model.eval()
    # GPU가 사용가능하면 GPU를 이용한 옵션으로 추론기 생성(파이토치에서 템플릿처럼 사용됨)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 입력 데이터 로드
    image_path = "images/demo/demo1.jpg"
    # 이미지 전처리(이미지 사이즈 transform)
    transform = get_transform(image_size=384)
    # CPU나 GPU 등 미리 설정된 divice 이용하도록 설정
    image = transform(Image.open(image_path)).unsqueeze(0).to(device)

    # STEP 4 : 추론 실행
    res = inference(image, model)

    # STEP 5 : 결과 후처리
    print("Image Tags: ", res[0])