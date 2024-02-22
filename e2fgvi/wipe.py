# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import importlib
import os
import argparse
from tqdm import tqdm
import torch
import subprocess

from pathlib import Path
from .core.utils import to_tensors

ffmpegExe = "/usr/local/ffmpeg/bin/ffmpeg"
ffprobeExe = "/usr/local/ffmpeg/bin/ffprobe"
parser = argparse.ArgumentParser(description="E2FGVI")
parser.add_argument("-v", "--video", type=str, required=True)
parser.add_argument("-c", "--ckpt", type=str, required=True)
parser.add_argument("--model", type=str, choices=['e2fgvi', 'e2fgvi_hq'])
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--num_ref", type=int, default=-1)
parser.add_argument("--neighbor_stride", type=int, default=10)
parser.add_argument("--savefps", type=int, default=24)

# frame_stride must be evenly divisible by neighbor_stride
parser.add_argument("--frame_stride", type=int, default=40)


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

args = parser.parse_args()

ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps


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

def create_mask(image, rect_top_left, rect_bottom_right, lower_color, upper_color):
    # 定义矩形区域的掩码
    if args.task == 'detext':
        rect_mask = np.zeros_like(image[:, :, 0])
        cv2.rectangle(rect_mask, rect_top_left, rect_bottom_right, (255, 255, 255), thickness=cv2.FILLED)
        # 应用矩形区域的掩码
        frame_modified = cv2.bitwise_and(image, image, mask=rect_mask)
        # 创建掩码
        mask = cv2.inRange(frame_modified, lower_color, upper_color)
        # 定义膨胀的核（kernel），这里使用矩形核
        kernel = np.ones((args.expand, args.expand), np.uint8)
        # 对掩码进行膨胀操作
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        frame_modified[dilated_mask > 0] = [255, 255, 255]
        frame_modified[dilated_mask <= 0] = [0, 0, 0]
    else:
        rect_mask = np.zeros_like(image[:, :, 0])
        cv2.rectangle(rect_mask, rect_top_left, rect_bottom_right, (255, 255, 255), thickness=cv2.FILLED)
        # 应用矩形区域的掩码
        frame_modified = cv2.bitwise_and(image, image, mask=rect_mask)
    return frame_modified
#  read frames from video
def read_frame_from_videos(npy_path, rect_top_left, rect_bottom_right):
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
    else:
        lower_color = np.array([int(i) for i in args.color_ranges[0].split('_')])
        upper_color = np.array([int(i)+1 for i in args.color_ranges[1].split('_')])
    if args.keep_mask:
        mask_path = Path(args.result) / f"{Path(args.video).stem}_{args.task}"
        if not mask_path.exists():
            mask_path.mkdir()
    print(rect_top_left, rect_bottom_right, lower_color, upper_color)
    if args.use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        while success:
            msec = frame_index * 1000 / default_fps
            is_excluded = False
            for r in ranges:
                if r[0] <= msec <= r[1]:
                    is_excluded = True
                    break
            if is_excluded:
                npy_file = npy_path / f'{frame_index}.npy'
                np.save(str(npy_file), image)
            else:
                frame_modified = create_mask(image, rect_top_left, rect_bottom_right, lower_color, upper_color)
                mask = Image.fromarray(cv2.cvtColor(frame_modified, cv2.COLOR_BGR2RGB))
                mask = np.array(mask.convert('L'))
                mask = np.array(mask > 0).astype(np.uint8)
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
                masks.append(Image.fromarray(mask * 255))
                if args.keep_mask:
                    # 生成文件名，例如：00000.jpg
                    mask_filename = mask_path / f"{frame_index:06d}.jpg"
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
                frame_modified = create_mask(image, rect_top_left, rect_bottom_right, lower_color, upper_color)
                mask = Image.fromarray(cv2.cvtColor(frame_modified, cv2.COLOR_BGR2RGB))
                mask = np.array(mask.convert('L'))
                mask = np.array(mask > 0).astype(np.uint8)
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
                masks.append(Image.fromarray(mask * 255))
                if args.keep_mask:
                    # 生成文件名，例如：00000.jpg
                    mask_filename = mask_path / f"{frame_index:06d}.jpg"
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

    top = max(0, args.box[0])
    bottom = min(height, args.box[1])
    left = max(0, args.box[2])
    right = min(width, args.box[3])
    # 矩形区域的左上角和右下角坐标
    rect_top_left = (top, left)
    rect_bottom_right = (bottom, right)

    video_path = str(Path(args.result) / f"{Path(args.video).stem}_{args.task}.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), default_fps, size)

    next_x_frames = None
    next_x_masks = None
    last_comp_frames = None
    framestride = args.frame_stride
    npy_path = Path(args.result) / f"{Path(args.video).stem}_npy"
    if not npy_path.exists():
        npy_path.mkdir()
    generator = read_frame_from_videos(npy_path, rect_top_left, rect_bottom_right)
    frame_index = 0
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
        h, w = size[1], size[0]

        print(f'Start test...')
        stride_length = len(x_frames)

        #strides = len(x_frames)

        #print(strides)
        #print(stride_length)

        #loopstartframe = 0
        #loopendframe = stride_length

        #xfram = x_frames[itern]
        #xmask = x_masks[itern]
        xfram = x_frames.copy()
        xmask = x_masks.copy()

        #if (itern < strides - 1):
        if next_x_frames is not None:
            for xframeppend in range(0, neighbor_stride):
                xfram.append(next_x_frames[xframeppend])
                xmask.append(next_x_masks[xframeppend])

        # if (itern > 0):
        #     for xframeppend in range(1, neighbor_stride + 1):
        #         xfram.insert(0, x_frames[itern - 1][len(x_frames[itern - 1]) - xframeppend])
        #         xmask.insert(0, x_masks[itern - 1][len(x_masks[itern - 1]) - xframeppend])

        imgs = to_tensors()(xfram).unsqueeze(0) * 2 - 1
        frames = [np.array(f).astype(np.uint8) for f in xfram]

        binary_masks = [
            np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in xmask
        ]
        masks = to_tensors()(xmask).unsqueeze(0)
        imgs, masks = imgs.half().to(device), masks.half().to(device)

        #if itern > 0:
        if last_comp_frames is not None:
            loopstartframe = neighbor_stride
        else:
            loopstartframe = 0

        #if (itern < strides - 1):
        if next_x_frames is not None:
            loopendframe = stride_length + neighbor_stride
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
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
        # saving videos
        video_length = (len(comp_frames)-neighbor_stride) if next_x_frames is not None else len(comp_frames)
        print('Saving videos...', video_length)
        for f in range(video_length):
            if comp_frames[f] is None:
                continue
            npy_file = npy_path / f'{frame_index}.npy'
            while npy_file.exists():
                frame = np.load(str(npy_file))
                writer.write(frame)
                npy_file.unlink()
                #print('-'*20, frame_index)
                frame_index += 1
                npy_file = npy_path / f'{frame_index}.npy'
            comp = comp_frames[f].astype(np.uint8)
            writer.write(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
            #print('-'*20, frame_index)
            frame_index += 1
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
    print(f'Finish test! The result video is saved in: {out_path}.')


if __name__ == '__main__':
    main_worker()
