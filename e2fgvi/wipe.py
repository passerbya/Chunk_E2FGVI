# -*- coding: utf-8 -*-

import sys
import cv2
import re
from PIL import Image
import numpy as np
import importlib
import os
import argparse
from tqdm import tqdm
import torch
import subprocess
import langid
import hashlib
import json
import shutil

from pathlib import Path
from .core.utils import to_tensors

ffmpegExe = "/usr/local/ffmpeg/bin/ffmpeg"
ffprobeExe = "/usr/local/ffmpeg/bin/ffprobe"
parser = argparse.ArgumentParser(description="E2FGVI")
parser.add_argument("-v", "--video", type=str, required=True)
parser.add_argument("-c", "--ckpt", type=str, required=True)
parser.add_argument("--model", type=str, choices=['e2fgvi', 'e2fgvi_hq'])
parser.add_argument("--step", type=int, default=5)
parser.add_argument("--num_ref", type=int, default=-1)
parser.add_argument("--neighbor_stride", type=int, default=5)
parser.add_argument("--savefps", type=int, default=24)

# frame_stride must be evenly divisible by neighbor_stride
parser.add_argument("--frame_stride", type=int, default=25)


# args for e2fgvi_hq (which can handle videos with arbitrary resolution)
parser.add_argument("--set_size", action='store_true', default=False)
parser.add_argument("--width", type=int)
parser.add_argument("--height", type=int)
parser.add_argument("-b", "--box", nargs='+', type=int,
                    help='Specify a mask box for the subtilte. Syntax: (top, bottom, left, right).')
parser.add_argument("--expand", type=int, default=6)
parser.add_argument("--exclude_ranges", nargs='+', type=str,
                    help='Specify exclude time ranges which not recognize. Syntax: (0_13880, 143360_146333).')
parser.add_argument( "--color_ranges", nargs='+', type=str,
                    help='Specify mask color ranges. Syntax: (230_230_230, 255_255_255).')
parser.add_argument("-r", "--result",  type=str, default='result/')
parser.add_argument("-t", "--task", type=str, help='CHOOSE THE TASK：delogo or detext', default='detext')
parser.add_argument("--keep_mask", action='store_true', default=False)
parser.add_argument("--sub_file", type=str)
parser.add_argument("--sub_offset", type=float, default=0.0)
parser.add_argument("--preview", type=int, default=-1)
parser.add_argument("--hard_codes", nargs='+', type=str, default='【硬】')

args = parser.parse_args()

ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps

def md5sum(s):
    m = hashlib.md5()
    m.update(s.encode('utf-8'))

    return m.hexdigest()

# sample reference frames from the whole video
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx = min(length, f + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index

def create_mask(image, rect, lower_color, upper_color):
    # 定义矩形区域的掩码
    if args.task == 'detext':
        rect_mask = np.zeros_like(image[:, :, 0])
        for rect_left_top, rect_right_bottom, _ in rect:
            cv2.rectangle(rect_mask, rect_left_top, rect_right_bottom, (255, 255, 255), thickness=cv2.FILLED)
        # 应用矩形区域的掩码
        frame_modified = cv2.bitwise_and(image, image, mask=rect_mask)
        # 创建掩码
        mask = cv2.inRange(frame_modified, lower_color, upper_color)
        # 定义膨胀的核（kernel），这里使用矩形核
        kernel = np.ones((args.expand, args.expand), np.uint8)
        # 对掩码进行膨胀操作
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        if len(frame_modified[dilated_mask > 0]) == 0:
            return None
        frame_modified[dilated_mask > 0] = [255, 255, 255]
        frame_modified[dilated_mask <= 0] = [0, 0, 0]
    else:
        rect_mask = np.zeros_like(image[:, :, 0])
        for rect_left_top, rect_right_bottom, _ in rect:
            cv2.rectangle(rect_mask, rect_left_top, rect_right_bottom, (255, 255, 255), thickness=cv2.FILLED)
        # 应用矩形区域的掩码
        frame_modified = cv2.bitwise_and(image, image, mask=rect_mask)
    return frame_modified

def get_previous_mask(rect, lower_color, upper_color, frame_index, frame_indexes):
    if frame_index in frame_indexes and len(frame_indexes[frame_index]) > 0:
        _frame_index = frame_index
        npy_path = Path(args.result) / f"{Path(args.video).stem}_npy_pre"
        if not npy_path.exists():
            npy_path.mkdir()
        f_index = 0
        vidcap = cv2.VideoCapture(args.video)
        start = max(0, frame_index-100)
        while True:
            success, frame = vidcap.read()
            if not success or f_index >= frame_index:
                break
            if f_index < start:
                f_index += 1
                continue
            np.save(str(npy_path/f'{f_index}.npy'), frame)
            f_index += 1
        vidcap.release()
        while frame_index-1>=start:
            frame_index-=1
            if frame_index not in frame_indexes:
                continue
            _rect = []
            for box in frame_indexes[frame_index]:
                _rect_left_top = (box[0], box[1])
                _rect_right_bottom = (box[2], box[3])
                _rect.append((_rect_left_top, _rect_right_bottom, box[4]))
            if len(rect) != len(_rect):
                return None
            for box, _box in zip([tup for tup in rect if len(tup[2]) > 0], [_tup for _tup in _rect if len(_tup[2]) > 0]):
                width_diff = abs((box[1][0]-box[0][0])-(_box[1][0]-_box[0][0]))
                height_diff = abs((box[1][1]-box[0][1])-(_box[1][1]-_box[0][1]))
                max_height = max(box[1][1]-box[0][1], _box[1][1]-_box[0][1])
                if width_diff > max_height/2 or height_diff > max_height/2 or box[2] != _box[2]:
                    #选区范围超过1/2个字体高度或者文本相似性小于0.8就认为是不同的文字
                    print(_frame_index, width_diff, height_diff, max_height/2, box[2], _box[2], frame_indexes[frame_index])
                    return None
                else:
                    frame = np.load(str(npy_path/f'{frame_index}.npy'))
                    frame_modified = create_mask(frame, _rect, lower_color, upper_color)
                    if frame_modified is not None:
                        return frame_modified
        shutil.rmtree(npy_path)
    return None

def get_next_mask(rect, lower_color, upper_color, frame_index, frame_indexes):
    if frame_index in frame_indexes and len(frame_indexes[frame_index]) > 0:
        _frame_index = frame_index
        f_index = 0
        vidcap = cv2.VideoCapture(args.video)
        while frame_index+1<len(frame_indexes):
            frame_index+=1
            if frame_index not in frame_indexes:
                continue
            _rect = []
            for box in frame_indexes[frame_index]:
                _rect_left_top = (box[0], box[1])
                _rect_right_bottom = (box[2], box[3])
                _rect.append((_rect_left_top, _rect_right_bottom, box[4]))
            if len(rect) != len(_rect):
                return None
            for box, _box in zip([tup for tup in rect if len(tup[2]) > 0], [_tup for _tup in _rect if len(_tup[2]) > 0]):
                width_diff = abs((box[1][0]-box[0][0])-(_box[1][0]-_box[0][0]))
                height_diff = abs((box[1][1]-box[0][1])-(_box[1][1]-_box[0][1]))
                max_height = max(box[1][1]-box[0][1], _box[1][1]-_box[0][1])
                if width_diff > max_height/2 or height_diff > max_height/2 or box[2] != _box[2]:
                    #选区范围超过1/2个字体高度或者文本相似性小于0.8就认为是不同的文字
                    print(_frame_index, width_diff, height_diff, max_height/2, box[2], _box[2], frame_indexes[frame_index])
                    return None
                else:
                    while f_index < frame_index:
                        success, frame = vidcap.read()
                        if not success:
                            break
                        f_index += 1
                    frame_modified = create_mask(frame, _rect, lower_color, upper_color)
                    if frame_modified is not None:
                        return frame_modified
        vidcap.release()
    return None

#  read frames from video
def read_frame_from_videos(npy_path, rect_left_top, rect_right_bottom, frame_indexes):
    frame_index = 0
    vname = args.video
    frames = []
    masks = []
    if args.exclude_ranges is None:
        ranges = []
    else:
        ranges = [[int(i) for i in r.split('_')] for r in args.exclude_ranges]
    if args.color_ranges is None:
        lower_color = None
        upper_color = None
        args.task = 'delogo'
    else:
        lower_color = np.array([int(i) for i in args.color_ranges[0].split('_')])
        upper_color = np.array([int(i)+1 for i in args.color_ranges[1].split('_')])
    if args.keep_mask:
        mask_path = Path(args.result) / f"{Path(args.video).stem}_{args.task}"
        if not mask_path.exists():
            mask_path.mkdir()

    if args.use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        while success:
            msec = frame_index * 1000 / default_fps
            is_excluded = False
            if len(frame_indexes) > 0:
                is_excluded = frame_index not in frame_indexes
            else:
                for r in ranges:
                    if r[0] <= msec <= r[1]:
                        is_excluded = True
                        break
            if is_excluded:
                npy_file = npy_path / f'{frame_index}.npy'
                np.save(str(npy_file), image)
            else:
                if frame_index in frame_indexes and len(frame_indexes[frame_index]) > 0:
                    rect = []
                    for box in frame_indexes[frame_index]:
                        _rect_left_top = (box[0], box[1])
                        _rect_right_bottom = (box[2], box[3])
                        rect.append((_rect_left_top, _rect_right_bottom, box[4]))
                else:
                    rect = [(rect_left_top, rect_right_bottom, '')]
                frame_modified = create_mask(image, rect, lower_color, upper_color)
                if frame_modified is None:
                    frame_modified = get_previous_mask(rect, lower_color, upper_color, frame_index, frame_indexes)
                if frame_modified is None:
                    frame_modified = get_next_mask(rect, lower_color, upper_color, frame_index, frame_indexes)
                if frame_modified is None:
                    npy_file = npy_path / f'{frame_index}.npy'
                    np.save(str(npy_file), image)
                    print('skip ', frame_index)
                else:
                    mask = Image.fromarray(cv2.cvtColor(frame_modified, cv2.COLOR_BGR2RGB))
                    mask = np.array(mask.convert('L'))
                    mask = np.array(mask > 0).astype(np.uint8)
                    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
                    masks.append((Image.fromarray(mask * 255), rect))
                    if args.keep_mask:
                        # 生成文件名，例如：00000.jpg
                        mask_filename = mask_path / f"{frame_index:07d}.jpg"
                        # 保存图像
                        cv2.imwrite(str(mask_filename), frame_modified)

                    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    frames.append(image)
                    if len(frames) >= args.frame_stride:
                        yield frames, masks
                        frames = []
                        masks = []
            success, image = vidcap.read()
            frame_index += 1
        vidcap.release()
    else:
        lst = os.listdir(vname)
        lst.sort()
        fr_lst = [vname + '/' + name for name in lst]
        for fr in fr_lst:
            image = cv2.imread(fr)
            msec = frame_index * 1000 / default_fps
            is_excluded = False
            for r in ranges:
                if r[0] <= msec <= r[1]:
                    is_excluded = True
                    break
            if is_excluded:
                npy_file = npy_path / f'{frame_index}.npy'
                np.save(str(npy_file), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                rect = [(rect_left_top, rect_right_bottom, '')]
                frame_modified = create_mask(image, rect, lower_color, upper_color)
                if frame_modified is None:
                    npy_file = npy_path / f'{frame_index}.npy'
                    np.save(str(npy_file), image)
                    print('skip ', frame_index)
                else:
                    mask = Image.fromarray(cv2.cvtColor(frame_modified, cv2.COLOR_BGR2RGB))
                    mask = np.array(mask.convert('L'))
                    mask = np.array(mask > 0).astype(np.uint8)
                    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
                    masks.append((Image.fromarray(mask * 255), rect))
                    if args.keep_mask:
                        # 生成文件名，例如：00000.jpg
                        mask_filename = mask_path / f"{frame_index:07d}.jpg"
                        # 保存图像
                        cv2.imwrite(str(mask_filename), frame_modified)

                    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    frames.append(image)
                    if len(frames) >= args.frame_stride:
                        yield frames, masks
                        frames = []
                        masks = []
            frame_index += 1
    if len(frames) > 0:
        yield frames, masks


# resize frames
def resize_frames(frames, size=None):
    if size is not None:
        frames = [f.resize(size) for f in frames]
    else:
        size = frames[0].size
    return frames, size

def check_file_has_audio(f):
    analysis_cmd = '%s -v quiet -select_streams a -show_entries stream=codec_name,channels,sample_rate -of default=nokey=1:noprint_wrappers=1 -i "%s"' % (ffprobeExe, f)
    try:
        probe_out = subprocess.check_output(analysis_cmd, shell=True).decode('utf-8', 'ignore')
        return len(probe_out) > 0 and probe_out.find('N/A') == -1
    except Exception:
        # logger.exception("")
        return False

def main_worker():
    # set up models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "e2fgvi":
        size = (432, 240)
    elif args.set_size:
        size = (args.width, args.height)
    else:
        size = None

    net = importlib.import_module('e2fgvi.model.' + args.model)
    model = net.InpaintGenerator().half().to(device)
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data)
    print(f'Loading model from: {args.ckpt}')
    model.eval()

    # prepare datset
    args.use_mp4 = True if args.video.endswith('.mp4') else False
    print(
        f'Loading videos and masks from: {args.video} | INPUT MP4 format: {args.use_mp4}'
    )

    if size is None:
        global default_fps
        video_stream = cv2.VideoCapture(args.video)
        default_fps = video_stream.get(cv2.CAP_PROP_FPS)
        _, frame = video_stream.read()
        height, width = frame.shape[:-1]
        size = (width, height)
        video_stream.release()
    if args.preview > 0 and args.use_mp4:
        preview_frame = args.preview * default_fps / 1000
        preview_begin = preview_frame - default_fps
        if preview_begin < 0:
            preview_begin = 0
        preview_begin = int(preview_begin * 1000 / default_fps)
        preview_path = str(Path(args.result) / f"{Path(args.video).stem}_preview_{preview_begin}_{preview_begin+2000}.mp4")
        command = '{} -ss {}ms -t 2 -i {} -y {}'.format(ffmpegExe, preview_begin, args.video, preview_path)
        print(command)
        subprocess.call(command, shell=True)
        args.video = preview_path

    top = max(0, args.box[0])
    bottom = min(height, args.box[1])
    left = max(0, args.box[2])
    right = min(width, args.box[3])
    # 矩形区域的左上角和右下角坐标
    rect_left_top = (left, top)
    rect_right_bottom = (right, bottom)

    frame_indexes = {}
    if args.sub_file and os.path.exists(args.sub_file):
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        with open(args.sub_file, 'r', encoding='utf-8') as file:
            load_subs = json.load(file)
        sorted_subs = []
        for sub1 in load_subs:
            is_hard_sub = False
            for hard_code in args.hard_codes:
                if sub1['txt'].startswith(hard_code):
                    is_hard_sub = True
                    break
            if is_hard_sub and 'wipe' not in sub1:
                continue
            sorted_subs.append(sub1)
        sub_offset = int(args.sub_offset*1000)
        sorted_subs.sort(key=lambda x:(x['st'], x['et']))
        sub_end = sorted_subs[-1]['et']
        for i1, sub in enumerate(sorted_subs):
            st1 = sub['st']
            et1 = sub['et']
            if i1 > 1:
                et0 = sorted_subs[i1-1]['et']
                if et0 < st1:
                    #前后两条字幕不重叠
                    while offset>0 and et0+offset>st1-offset:
                        #原本不重叠扩大时间轴后保持不重叠
                        offset -= 1
                elif et0 == st1:
                    offset = 0
                sub['st'] = (st1-offset) if (st1-offset)>0 else 0
            if i1 < len(sorted_subs)-1:
                st2 = sorted_subs[i1+1]['st']
                offset = sub_offset
                if et1 < st2:
                    #前后两条字幕不重叠
                    while offset>0 and et1+offset>st2-offset:
                        #原本不重叠扩大时间轴后保持不重叠
                        offset -= 1
                elif et1 == st2:
                    offset = 0
                sub['et'] = (et1+offset) if (et1+offset)<sub_end else sub_end
        sorted_subs.sort(key=lambda x:(x['st'], x['et']))
        timestamps = []
        all_sub_txt = ''
        sep = ''
        for sub in sorted_subs:
            timestamps.append(sub['st'])
            timestamps.append(sub['et'])
            txt = sub['txt']
            txt = re.sub('[\r\n]+', ' ', txt)
            sub['txt'] = txt
            for hard_code in args.hard_codes:
                if txt.startswith(hard_code):
                    txt = txt[len(hard_code):]
            all_sub_txt += sep + txt
            sep = ' '
        lang = langid.classify(all_sub_txt)[0]
        print(lang)

        timestamps = list(set(timestamps))
        timestamps.sort()
        times_len = len(timestamps)
        subs = []
        for i in range(times_len-1):
            st = timestamps[i]
            et = timestamps[i+1]
            children = []
            sub = {'st':st,'et':et,'subs':children}
            is_all_not_wipe = True
            for sub1 in sorted_subs:
                st1 = sub1['st']
                et1 = sub1['et']
                txt1 = sub1['txt']
                is_hard_sub = False
                for hard_code in args.hard_codes:
                    if txt1.startswith(hard_code):
                        txt1 = txt1[len(hard_code):]
                        is_hard_sub = True
                wipe1 = sub1['wipe'] if 'wipe' in sub1 else 1
                if st1 <= st and et1 >= et:
                    if wipe1 == 1:
                        is_all_not_wipe = False
                    children.append({'txt':txt1,'wipe':wipe1,'is_hard_sub':is_hard_sub})
                if st1 > et:
                    break
            if not is_all_not_wipe:
                subs.append(sub)

        frame_index = 0
        vidcap = cv2.VideoCapture(args.video)
        for sub in subs:
            st = sub['st']
            et = sub['et']
            has_hard_sub = False
            all_sub_txt = ''
            sep = ''
            for child in sub['subs']:
                txt = child['txt']
                if child['is_hard_sub']:
                    has_hard_sub = True
                elif child['wipe'] == 1:
                    all_sub_txt += sep + txt
                    sep = ' '

            frame_dir = Path(args.result) / f"{st}_{et}/"
            if frame_dir.exists():
                for img in frame_dir.glob("*.jpg"):
                    img.unlink()
            else:
                frame_dir.mkdir()
            while True:
                success, frame = vidcap.read()
                if not success:
                    break
                msec = frame_index * 1000 / default_fps
                if msec < st:
                    #print('-'*20, st, msec, et, frame_index)
                    frame_index += 1
                    continue
                #print(st, msec, et, frame_index, has_hard_sub)
                if  msec < et:
                    frame_indexes[frame_index] = []
                    cv2.imwrite(str(frame_dir / f'{frame_index}.jpg'), frame)
                frame_index += 1
                if (frame_index * 1000 / default_fps) >= et:
                    break
            img_stems = []
            for img in frame_dir.glob("*.jpg"):
                img_stems.append(int(img.stem))
            img_stems.sort()
            if has_hard_sub:
                keep_hard_sub = False
                for child in sub['subs']:
                    txt = child['txt']
                    if not child['is_hard_sub']:
                        continue
                    content_md5 = md5sum(txt)
                    txt = txt.replace("'", "\'")
                    if child['wipe'] == 1:
                        if lang in ('bn','th','fa','ug','ur','ps','so'):
                            cmd = f"source /etc/profile_easyocr && python3 {str(script_dir/'ocr_easyocr.py')} --lang {lang} --frame_dir {str(frame_dir)} --content '{txt}'"
                        else:
                            cmd = f"source /etc/profile_paddle && python3 {str(script_dir/'ocr_paddle.py')} --lang {lang} --frame_dir {str(frame_dir)} --content '{txt}'"
                        print(cmd)
                        subprocess.check_output(cmd, shell=True).decode('utf-8', 'ignore')
                        with open(str(Path(frame_dir) / f'{content_md5}_box.json'), 'r', encoding='utf-8') as file:
                            child['boxes'] = json.load(file)
                    else:
                        keep_hard_sub = True

                if keep_hard_sub:
                    content_md5 = md5sum(all_sub_txt)
                    all_sub_txt = all_sub_txt.replace("'", "\'")
                    if lang in ('bn','th','fa','ug','ur','ps','so'):
                        cmd = f"source /etc/profile_easyocr && python3 {str(script_dir/'ocr_easyocr.py')} --lang {lang} --frame_dir {str(frame_dir)} --content '{all_sub_txt}' --box {top} {bottom} {left} {right}"
                    else:
                        cmd = f"source /etc/profile_paddle && python3 {str(script_dir/'ocr_paddle.py')} --lang {lang} --frame_dir {str(frame_dir)} --content '{all_sub_txt}' --box {top} {bottom} {left} {right}"
                    print(cmd)
                    subprocess.check_output(cmd, shell=True).decode('utf-8', 'ignore')
                    with open(str(Path(frame_dir) / f'{content_md5}_box.json'), 'r', encoding='utf-8') as file:
                        boxes = json.load(file)
                    for i1, img_stem in enumerate(img_stems):
                        box = boxes[i1]
                        if box[0] > 0.8:
                            box[1].append(all_sub_txt)
                            frame_indexes[img_stem] = [box[1]]
                        else:
                            last_box = boxes[-1][1]
                            if last_box[0] == last_box[1] == last_box[2] == last_box[3] == 0:
                                frame_indexes[img_stem] = [(left, top, right, bottom, '')]
                            else:
                                boxes[-1][1].append(all_sub_txt)
                                frame_indexes[img_stem] = [boxes[-1][1]]
                else:
                    for img_stem in img_stems:
                        frame_indexes[img_stem] = [(left, top, right, bottom, '')]

                for child in sub['subs']:
                    if 'boxes' not in child:
                        continue
                    boxes = child['boxes']
                    for i1, img_stem in enumerate(img_stems):
                        box = boxes[i1]
                        if box[0] > 0.8:
                            box = box[1]
                        else:
                            box = boxes[-1][1]
                        if box[0] == box[1] == box[2] == box[3] == 0:
                            continue
                        box.append(child['txt'])
                        frame_indexes[img_stem].append(box)
            else:
                for img_stem in img_stems:
                    frame_indexes[img_stem] = [(left, top, right, bottom, '')]

        vidcap.release()
    video_path = str(Path(args.result) / f"{Path(args.video).stem}_{args.task}.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), default_fps, size)

    next_x_frames = None
    next_x_masks = None
    last_comp_frames = None
    framestride = args.frame_stride
    npy_path = Path(args.result) / f"{Path(args.video).stem}_npy"
    if not npy_path.exists():
        npy_path.mkdir()

    generator = read_frame_from_videos(npy_path, rect_left_top, rect_right_bottom, frame_indexes)
    frame_index = 0
    crop_top = max(0, args.box[0]-20)
    crop_bottom = min(height, args.box[1]+20)
    crop_left = max(0, args.box[2]-20)
    crop_right = min(width, args.box[3]+20)
    while True:
        if next_x_frames is not None:
            x_frames = next_x_frames
            x_masks = next_x_masks
            last_comp_frames = comp_frames[framestride:]
            comp_frames = [None] * (len(x_frames) + neighbor_stride)
            for idx in range(len(last_comp_frames)):
                comp_frames[idx] = last_comp_frames[idx]
        else:
            try:
                x_frames, x_masks = next(generator)
            except StopIteration:
                break
            comp_frames = [None] * (framestride + neighbor_stride)
        try:
            next_x_frames, next_x_masks = next(generator)
        except StopIteration:
            next_x_frames = None
            next_x_masks = None
            pass

        x_frames, size = resize_frames(x_frames, size)
        if next_x_frames is not None:
            next_x_frames, _ = resize_frames(next_x_frames, size)
        #h, w = size[1], size[0]

        print(f'Start wipe...')
        stride_length = len(x_frames)

        xfram = x_frames.copy()
        xmask = x_masks.copy()

        if next_x_frames is not None:
            for xframeppend in range(0, min(len(next_x_frames), neighbor_stride)):
                xfram.append(next_x_frames[xframeppend])
                xmask.append(next_x_masks[xframeppend])

        _crop_left = crop_left
        _crop_top = crop_top
        _crop_right = crop_right
        _crop_bottom = crop_bottom
        for xm in xmask:
            for mask_left_top, mask_right_bottom, _ in xm[1]:
                _crop_left = min(_crop_left, mask_left_top[0])
                _crop_top = min(_crop_top, mask_left_top[1])
                _crop_right = max(_crop_right, mask_right_bottom[0])
                _crop_bottom = max(_crop_bottom, mask_right_bottom[1])
        #print((crop_left, crop_top, crop_right, crop_bottom), (_crop_left, _crop_top, _crop_right, _crop_bottom))

        _xfram = [xf.crop((_crop_left, _crop_top, _crop_right, _crop_bottom)) for xf in xfram]
        _xmask = [xm[0].crop((_crop_left, _crop_top, _crop_right, _crop_bottom)) for xm in xmask]
        w, h = _xfram[0].size
        imgs = to_tensors()(_xfram).unsqueeze(0) * 2 - 1
        frames = [np.array(f).astype(np.uint8) for f in _xfram]

        binary_masks = [
            np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in _xmask
        ]
        masks = to_tensors()(_xmask).unsqueeze(0)
        imgs, masks = imgs.half().to(device), masks.half().to(device)

        #if itern > 0:
        if last_comp_frames is not None:
            loopstartframe = len(last_comp_frames)
        else:
            loopstartframe = 0

        #if (itern < strides - 1):
        if next_x_frames is not None:
            loopendframe = stride_length + min(len(next_x_frames), neighbor_stride)
        else:
            loopendframe = stride_length

        # completing holes by e2fgvi

        #print(loopstartframe, loopendframe, neighbor_stride)
        for f in tqdm(range(loopstartframe, loopendframe, neighbor_stride)):
            #print(f)
            #print(f'meh {max(loopstartframe, f - neighbor_stride)} muh {min(loopendframe, f + neighbor_stride + 1)}')
            neighbor_ids = [
                i for i in range(max(loopstartframe, f - neighbor_stride),
                                 min(loopendframe, f + neighbor_stride + 1))
            ]
            # The frame +- 5 frames before or after.  Beginning: zero frames before.   End: 0 frames after.

            ref_ids = get_ref_index(f, neighbor_ids, loopendframe)
            selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]

            with torch.no_grad():
                masked_imgs = selected_imgs * (1 - selected_masks)
                mod_size_h = 60
                mod_size_w = 108
                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
                masked_imgs = torch.cat(
                    [masked_imgs, torch.flip(masked_imgs, [3])],
                    3)[:, :, :, :h + h_pad, :]
                masked_imgs = torch.cat(
                    [masked_imgs, torch.flip(masked_imgs, [4])],
                    4)[:, :, :, :, :w + w_pad].half()
                pred_imgs, _ = model(masked_imgs, len(neighbor_ids))
                pred_imgs = pred_imgs[:, :, :h, :w]
                pred_imgs = (pred_imgs + 1) / 2
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
                for i in range(0, len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_imgs[i]).astype(
                        np.uint8) * binary_masks[idx] + frames[idx] * (
                                  1 - binary_masks[idx])
                    xf = np.array(xfram[idx].convert("RGB"))
                    xf[_crop_top:_crop_bottom, _crop_left:_crop_right] = img
                    if comp_frames[idx] is None:
                        comp_frames[idx] = xf
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + xf.astype(np.float32) * 0.5
        # saving videos
        video_length = len(comp_frames) - neighbor_stride
        print('Saving videos...', video_length)
        for f in range(video_length):
            if comp_frames[f] is None:
                continue
            npy_file = npy_path / f'{frame_index}.npy'
            while npy_file.exists():
                frame = np.load(str(npy_file))
                writer.write(frame)
                npy_file.unlink()
                frame_index += 1
                npy_file = npy_path / f'{frame_index}.npy'
            comp = comp_frames[f].astype(np.uint8)
            comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
            writer.write(comp)
            frame_index += 1

    npy_file = npy_path / f'{frame_index}.npy'
    while npy_file.exists():
        frame = np.load(str(npy_file))
        writer.write(frame)
        npy_file.unlink()
        frame_index += 1
        npy_file = npy_path / f'{frame_index}.npy'
    writer.release()

    out_path = str(Path(args.result) / f"{Path(args.video).stem}_out.mp4")
    if check_file_has_audio(args.video):
        analysis_cmd = '%s -v quiet -of default=nokey=1:noprint_wrappers=1 -select_streams a -show_entries stream=bit_rate -i "%s"' % (ffprobeExe, args.video)
        a_out = subprocess.check_output(analysis_cmd, shell=True).decode('utf-8', 'ignore')

        analysis_cmd = '%s -v quiet -of default=nokey=1:noprint_wrappers=1 -select_streams v -show_entries stream=bit_rate -i "%s"' % (ffprobeExe, video_path)
        v_out = subprocess.check_output(analysis_cmd, shell=True).decode('utf-8', 'ignore')
        command = '{} -i {} -i {} -map 0:a -map 1:v -b:v {} -b:a {} -y {}'.format(ffmpegExe, args.video, video_path, v_out.strip(), a_out.strip(), out_path)
        subprocess.call(command, shell=True)
    else:
        os.rename(video_path, out_path)
    print(f'Finish wipe! The result video is saved in: {out_path}.', frame_index)


if __name__ == '__main__':
    main_worker()
