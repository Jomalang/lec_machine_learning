'''
 * The Tag2Text Model
 * Written by Xinyu Huang
'''
#STEP 1 : import modules
import argparse
import numpy as np
import random

import torch

from PIL import Image
from ram.models import tag2text
from ram import inference_tag2text as inference
from ram import get_transform


# 이하 코드는 터미널 환경에서 args이용하는 코드로, 주석처리
# parser = argparse.ArgumentParser(
#     description='Tag2Text inferece for tagging and captioning')
# parser.add_argument('--image',
#                     metavar='DIR',
#                     help='path to dataset',
#                     default='images/1641173_2291260800.jpg')
# parser.add_argument('--pretrained',
#                     metavar='DIR',
#                     help='path to pretrained model',
#                     default='pretrained/tag2text_swin_14m.pth')
# parser.add_argument('--image-size',
#                     default=384,
#                     type=int,
#                     metavar='N',
#                     help='input image size (default: 384)')
# parser.add_argument('--thre',
#                     default=0.68,
#                     type=float,
#                     metavar='N',
#                     help='threshold value')
# parser.add_argument('--specified-tags',
#                     default='None',
#                     help='User input specified tags')


if __name__ == "__main__":

    # args = parser.parse_args()

    #STEP2 : 추론기 객체 생성(옵션 설정)
    # 파이토치에서 사용할 device 설정(cpu or gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # delete some tags that may disturb captioning
    # 127: "quarter"; 2961: "back", 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
    delete_tag_index = [127,2961, 3351, 3265, 3338, 3355, 3359]
    #######load model
    model_path = "pretrained/tag2text_swin_14m.pth"
    model = tag2text(pretrained=model_path,
                             image_size=384,
                             vit='swin_b',
                             delete_tag_index=delete_tag_index)
    model.threshold = 0.68  # threshold for tagging
    model.eval()

    model = model.to(device)

    #STEP3 : 입력 데이터 로드
    image_path = "images/1641173_2291260800.jpg"
    transform = get_transform(image_size=384)
    image = transform(Image.open(image_path)).unsqueeze(0).to(device)


    #STEP4 : 추론 실행
    res = inference(image, model, 'None')


    #STEP5 : 반환값 후처리
    print("Model Identified Tags: ", res[0])
    print("User Specified Tags: ", res[1])
    print("Image Caption: ", res[2])