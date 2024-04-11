#!/usr/bin/python
# coding: utf-8
import sys
import os
import re
import json
import argparse
import Levenshtein
import hashlib
import cv2
from pathlib import Path
import logging.handlers
from paddleocr import PaddleOCR

def md5sum(s):
    m = hashlib.md5()
    m.update(s.encode('utf-8'))

    return m.hexdigest()

def is_chinese(content):
    mobj = re.search('[\u4E00-\u9FA5]+', content)
    return mobj is not None

def apply_similarity(s1, s2):
    distance = Levenshtein.distance(s1,s2)
    m = max(len(s1), len(s2))
    return (m-distance)/m

def recognize(lang, frame_dir, content, sub_box=None):
    os.environ["FLAGS_allocator_strategy"] = 'naive_best_fit'
    os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = '0.01'
    os.environ["FLAGS_gpu_memory_limit_mb"] = '2048'
    if lang == 'ch':
        ocr = PaddleOCR(det_model_dir='/root/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_server_infer', rec_model_dir='/root/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_server_infer', use_angle_cls=False, use_gpu=True, lang=lang)
    elif lang == 'en':
        ocr = PaddleOCR(det_model_dir='/root/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_server_infer', rec_model_dir='/root/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_server_infer', use_angle_cls=False, use_gpu=True, lang='ch')
    else:
        ocr = PaddleOCR(use_angle_cls=False, use_gpu=True, lang=lang)

    content = content.lower()
    img_dir = Path(frame_dir)
    logger.info('开始识别%s %s', img_dir, lang)
    imgs = []
    temp_file_path = None
    if sub_box is not None and len(sub_box) > 0:
        temp_file_path = img_dir / 'temp'
        if not temp_file_path.exists():
            temp_file_path.mkdir()
    width = 0
    for img in img_dir.glob("*.jpg"):
        if temp_file_path is not None:
            image = cv2.imread(str(img))
            img = temp_file_path / img.name
            cv2.imwrite(str(img), image[sub_box[0]:sub_box[1], sub_box[2]:sub_box[3]])
            width = sub_box[3] - sub_box[2]
        else:
            image = cv2.imread(str(img))
            _, width = image.shape[:-1]
        imgs.append((int(img.stem), img))
    imgs.sort(key=lambda x:x[0])
    boxes = []
    for img in imgs:
        result_list = ocr.ocr(str(img[1]), cls=False)
        if len(result_list) == 0 or result_list[0] is None:
            boxes.append((0, [0, 0, 0, 0]))
            continue
        result_list = result_list[0]
        #logger.info(result_list)
        size = len(result_list)
        rows = []
        word_index = set()
        for i in range(size):
            wr1 = result_list[i]
            if id(wr1) in word_index:
                continue
            j = i + 1
            y11 = wr1[0][0][1]
            y12 = wr1[0][2][1]
            columns = [wr1]
            word_index.add(id(wr1))
            while j < size:
                wr2 = result_list[j]
                y21 = wr2[0][0][1]
                y22 = wr2[0][2][1]
                if abs(y21 - y11) < min(y12 - y11, y22 - y21)/2 and abs(y22 - y12) < min(y12 - y11, y22 - y21)/2:
                    columns.append(wr2)
                    word_index.add(id(wr2))
                j += 1
            rows.append(columns)

        _rows = []
        all_txt = ''
        line_sep = ''
        for columns in rows:
            columns.sort(key=lambda x:x[0][0][0])
            txt = ''
            word_sep = ''
            for wr in columns:
                txt += word_sep + wr[1][0].lower()
                if lang not in ('ch', 'chinese_cht', 'japan'):
                    word_sep = ' '
                elif lang == 'ch' and not is_chinese(wr[1][0]):
                    #识别到英文
                    word_sep = ' '
            s = apply_similarity(txt, content)
            s1 = apply_similarity(all_txt, content) if len(all_txt) > 0 else 0
            s2 = apply_similarity(all_txt + line_sep + txt, content)
            if s > 0 and (s2 > s1 or (0.1 < s1 - s2 < 0)):
                #文字越多相似性越高，或者由于误差原因相似性略有下降
                all_txt += line_sep + txt
                line_sep = ' '
                _rows.append(columns)
        if len(all_txt) > 0:
            s = apply_similarity(all_txt, content)
            left = top = sys.maxsize
            right = bottom = 0
            for columns in _rows:
                for wr in columns:
                    for w in wr[0]:
                        if w[0] < left:
                            left = w[0]
                        if w[0] > right:
                            right = w[0]
                        if w[1] < top:
                            top = w[1]
                        if w[1] > bottom:
                            bottom = w[1]
                        offset = bottom - top
                        left = (left - offset) if (left - offset)>0 else 0
                        right = (right + offset) if (right + offset)<width else width
            logger.info('%s %s %s', s, (left, top, right, bottom), all_txt)
            boxes.append((s, [int(left), int(top), int(right), int(bottom)]))
        else:
            boxes.append((0, [0, 0, 0, 0]))

    _boxes = []
    for box in boxes:
        if box[0] < 0.7:
            continue
        _boxes.append(box)

    if len(_boxes) > 0:
        left = top = sys.maxsize
        right = bottom = 0
        for box in _boxes:
            if box[1][0] < left:
                left = box[1][0]
            if box[1][2] > right:
                right = box[1][2]
            if box[1][1] < top:
                top = box[1][1]
            if box[1][3] > bottom:
                bottom = box[1][3]
        boxes.append((1, [left, top, right, bottom]))
    if temp_file_path is not None:
        for box in boxes:
            if box[1] == [0, 0, 0, 0]:
                continue
            box[1][0] += sub_box[2]
            box[1][2] += sub_box[2]
            box[1][1] += sub_box[0]
            box[1][3] += sub_box[0]
    with open(str(Path(frame_dir) / f'{md5sum(content)}_box.json'), 'w', encoding='utf-8') as file:
        json.dump(boxes, file)
    logger.info(len(boxes))
    logger.info('识别%s %s结束', img_dir, lang)

logging.basicConfig()
logger = logging.getLogger("ocr_paddle")
formatter = logging.Formatter('%(asctime)s %(threadName)s %(funcName)s:%(lineno)d %(levelname)-8s: %(message)s')
file_handler = logging.handlers.RotatingFileHandler("/usr/local/data/ysj/logs/ocr_paddle.log", maxBytes=10 * 1024 * 1024, backupCount=5)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str)
    parser.add_argument('--frame_dir', type=str)
    parser.add_argument('--content', type=str)
    parser.add_argument("--box", nargs='+', type=int,
                        help='Specify a mask box for the subtilte. Syntax: (top, bottom, left, right).')
    args = parser.parse_args()
    #参数转换
    _lang = args.lang
    latin_lang_list = ['az','bs','hr','mt','es','fr','pt','de','it','nl','sv',
                       'da','no','fi','pl','cs','sk','hu','ro','tr','id','ms','vi',
                       'tl','sw','sq','et','lv','lt','ca','cy','ga','is','sl']
    arabic_lang_list = ['ar','fa','ug','ur','ps','so']
    cyrillic_lang_list = ['ru','uk','bg','mk','sr','mn','kg','tg','ka','uz']
    devanagari_lang_list = ['hi','ne','mr','si']

    if args.lang == 'zh':
        _lang = 'ch'
    elif args.lang == 'ja':
        _lang = 'japan'
    elif args.lang == 'ko':
        _lang = 'korean'
    elif args.lang in ('en', 'ta', 'te'):
        _lang = args.lang
    elif args.lang == 'kn':
        _lang = 'ka'
    elif args.lang in latin_lang_list:
        _lang = 'latin'
    elif args.lang in arabic_lang_list:
        _lang = 'arabic'
    elif args.lang in cyrillic_lang_list:
        _lang = 'cyrillic'
    elif args.lang in devanagari_lang_list:
        _lang = 'devanagari'

    recognize(_lang, args.frame_dir, args.content, args.box)
    '''
    recognize('en', '/usr/local/data/jtubespeech/0_1', 'to be continued')
    recognize('en', '/usr/local/data/jtubespeech/1_1', 'LOVING YOU IN SECRET 01')
    
    recognize('en', '0_1', '剧情纯属虚构 无不良引导 请勿模仿 树立正确价值观')
    recognize('en', '0_1', '我赶走了陈正豪')
    '''
