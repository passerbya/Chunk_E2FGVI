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
from e2fgvi.core.utils import to_tensors

ffmpegExe = "/usr/local/ffmpeg/bin/ffmpeg"
ffprobeExe = "/usr/local/ffmpeg/bin/ffprobe"
parser = argparse.ArgumentParser(description="E2FGVI")
parser.add_argument("-v", "--video", type=str, required=True)
parser.add_argument("-c", "--ckpt", type=str, required=True)
parser.add_argument("-m", "--mask", type=str, required=True)
parser.add_argument("--model", type=str, choices=['e2fgvi', 'e2fgvi_hq'])
parser.add_argument("--step", type=int, default=8)
parser.add_argument("--num_ref", type=int, default=-1)
parser.add_argument("--neighbor_stride", type=int, default=8)
parser.add_argument("--savefps", type=int, default=24)

# frame_stride must be evenly divisible by neighbor_stride
parser.add_argument("--frame_stride", type=int, default=40)


# args for e2fgvi_hq (which can handle videos with arbitrary resolution)
parser.add_argument("--set_size", action='store_true', default=False)
parser.add_argument("--width", type=int)
parser.add_argument("--height", type=int)

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


# read frame-wise masks
def read_mask(mpath, size, frame_stride):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for mp in mnames:
        m = Image.open(os.path.join(mpath, mp))
        m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                       iterations=4)
        masks.append(Image.fromarray(m * 255))
        if len(masks) >= frame_stride:
            yield masks
            masks = []
    if len(masks) > 0:
        yield masks


#  read frames from video
def read_frame_from_videos(mpath, npy_path):
    frame_index = 0
    vname = args.video
    frames = []
    if args.use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        while success:
            pic_path = Path(mpath) / f"{frame_index:05d}.jpg"
            mask = cv2.imread(str(pic_path))
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            if np.all(gray_mask == 0):
                npy_file = npy_path / f'{frame_index}.npy'
                np.save(str(npy_file), image)
                print('skip ', frame_index)
            else:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                frames.append(image)
                if len(frames) >= args.frame_stride:
                    yield frames
                    frames = []
            success, image = vidcap.read()
            frame_index += 1
        vidcap.release()
    else:
        lst = os.listdir(vname)
        lst.sort()
        fr_lst = [vname + '/' + name for name in lst]
        for fr in fr_lst:
            image = cv2.imread(fr)
            pic_path = Path(mpath) / f"{frame_index:05d}.jpg"
            mask = cv2.imread(str(pic_path))
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            if np.all(gray_mask == 0):
                npy_file = npy_path / f'{frame_index}.npy'
                np.save(str(npy_file), image)
                print('skip ', frame_index)
            else:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                frames.append(image)
                if len(frames) >= args.frame_stride:
                    yield frames
                    frames = []
            frame_index += 1
    if len(frames) > 0:
        yield frames

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

    video_path = str(Path('results') / f"{Path(args.video).stem}_results.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), default_fps, size)

    next_x_frames = None
    next_x_masks = None
    last_comp_frames = None
    framestride = args.frame_stride
    npy_path = Path('results') / f"{Path(args.video).stem}_npy"
    if not npy_path.exists():
        npy_path.mkdir()
    rframes = read_frame_from_videos(args.mask, npy_path)
    rmasks = read_mask(args.mask, size, framestride)
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
                x_frames = next(rframes)
                x_masks = next(rmasks)
            except StopIteration:
                break
            comp_frames = [None] * (framestride + neighbor_stride)
        try:
            next_x_frames = next(rframes)
        except StopIteration:
            next_x_frames = None
            pass
        try:
            next_x_masks = next(rmasks)
        except StopIteration:
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
            for xframeppend in range(0, min(len(next_x_frames), neighbor_stride)):
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
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
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
            writer.write(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
            frame_index += 1

    npy_file = npy_path / f'{frame_index}.npy'
    while npy_file.exists():
        frame = np.load(str(npy_file))
        writer.write(frame)
        npy_file.unlink()
        frame_index += 1
        npy_file = npy_path / f'{frame_index}.npy'
    writer.release()
    out_path = str(Path('results') / f"{Path(args.video).stem}_out.mp4")
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
