#!/usr/bin/python
# coding: utf-8

import sys
import os
import json
import argparse
import Levenshtein
from pathlib import Path
import easyocr
import hashlib
import logging.handlers

def md5sum(s):
    m = hashlib.md5()
    m.update(s.encode('utf-8'))

    return m.hexdigest()

def apply_similarity(s1, s2):
    distance = Levenshtein.distance(s1,s2)
    m = max(len(s1), len(s2))
    return (m-distance)/m

def recognize(lang, frame_dir, content):
    content = content.lower()
    img_dir = Path(frame_dir)
    logger.info('开始识别%s %s', img_dir, lang)
    imgs = []
    for img in img_dir.glob("*.jpg"):
        imgs.append((int(img.stem), img))
    imgs.sort(key=lambda x:x[0])
    boxes = []
    reader = easyocr.Reader(lang)
    for img in imgs:
        result_list = reader.readtext(str(img[1]))
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
                txt += word_sep + wr[1].lower()
                if lang[0] not in ('ch_sim', 'ch_tra', 'ja', 'th'):
                    word_sep = ' '
                else:
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
            logger.info('%s %s %s', s, (left, top, right, bottom), all_txt)
            boxes.append((s, (int(left), int(top), int(right), int(bottom))))
        else:
            boxes.append((0, (0, 0, 0, 0)))

    _boxes = []
    for box in boxes:
        if box[0] < 0.7:
            continue
        _boxes.append(box)

    left = top = sys.maxsize
    right = bottom = 0
    if len(_boxes) > 0:
        for box in _boxes:
            if box[1][0] < left:
                left = box[1][0]
            if box[1][2] > right:
                right = box[1][2]
            if box[1][1] < top:
                top = box[1][1]
            if box[1][3] > bottom:
                bottom = box[1][3]
        boxes.append((1, (left, top, right, bottom)))
    with open(str(Path(frame_dir) / f'{md5sum(content)}_box.json'), 'w', encoding='utf-8') as file:
        json.dump(boxes, file)
    logger.info(len(boxes))
    logger.info('识别%s %s结束', img_dir, lang)

logging.basicConfig()
logger = logging.getLogger("ocr_easyocr")
formatter = logging.Formatter('%(asctime)s %(threadName)s %(funcName)s:%(lineno)d %(levelname)-8s: %(message)s')
file_handler = logging.handlers.RotatingFileHandler("/usr/local/data/ysj/logs/ocr_easyocr.log", maxBytes=10 * 1024 * 1024, backupCount=5)
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
    args = parser.parse_args()
    #参数转换
    _lang = args.lang
    bengali_lang_list = ['bn','as','mni']
    latin_lang_list = ['az','bs','hr','mt','es','fr','pt','de','it','nl','sv',
                       'da','no','fi','pl','cs','sk','hu','ro','tr','id','ms','vi',
                       'tl','sw','sq','et','lv','lt','ca','cy','ga','is','sl']
    arabic_lang_list = ['ar','fa','ug','ur','ps','so']
    cyrillic_lang_list = ['ru','uk','bg','mk','sr','mn','kg','tg','ka','uz']
    devanagari_lang_list = ['hi','ne','mr','si']

    if args.lang == 'zh':
        _lang = ['ch_sim', 'en']
    elif args.lang == 'ja':
        _lang = ['ja', 'en']
    elif args.lang == 'ko':
        _lang = ['ko', 'en']
    elif args.lang in ('th', 'ta', 'te', 'kn'):
        _lang = [args.lang, 'en']
    elif args.lang in bengali_lang_list:
        _lang = ['bn', 'en']
    elif args.lang in arabic_lang_list:
        _lang = ['ar', 'en']
    elif args.lang in devanagari_lang_list:
        _lang = ['hi', 'en']
    elif args.lang in cyrillic_lang_list:
        _lang = ['ru', 'en']
    else:
        _lang = ['la']

    recognize(_lang, args.frame_dir, args.content)
    '''
    recognize(['ch_sim', 'en'], '/usr/local/data/jtubespeech/0_1', 'to be continued')
    recognize(['ch_sim', 'en'], '/usr/local/data/jtubespeech/1_1', 'LOVING YOU IN SECRET 01')
    
    recognize(['ch_sim', 'en'], '0_1', '剧情纯属虚构 无不良引导 请勿模仿 树立正确价值观')
    recognize(['ch_sim', 'en'], '0_1', '我赶走了陈正豪')
    '''
