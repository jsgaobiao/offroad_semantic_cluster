# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

RAW_DATA_FILE='0.avi'
OUTPUT_DIR='anchor_annotation'

flag_playing_video = False # 是否自动播放视频
flag_next_frame = True  # 播放下一帧视频
resize_ratio = 2.0
waitkey_time = 0
anchor_dict = {}
anchor_type = 0
anchor_width = 64
anchor_color = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0)]

# 保存已标注的结果
def save_anchor_to_file():
    np.save(os.path.join(OUTPUT_DIR, RAW_DATA_FILE+"_anchors_annotation.npy"), anchor_dict)

# 保存已标注的锚点对应的图像
def save_anchor_img_to_file():
    global anchor_dict, frame_id, cap
    for _anch in anchor_dict.keys():
        _img_file = os.path.join(OUTPUT_DIR, str(_anch)+'.png')
        if not os.path.isfile(_img_file):
            cap.set(cv2.CAP_PROP_POS_FRAMES, _anch-1)
            _ret, _img = cap.read()
            if _ret:
                cv2.imwrite(_img_file, _img)
                print('saving image file '+_img_file)
    print('Done.')
            
# 锚点标注的鼠标回调函数
def label_anchor_mouse_callback(event, x, y, flags, param):
    global anchors, anchor_type, anchor_dict
    global frame, frame_id, frame_with_anchor
    if event == cv2.EVENT_LBUTTONDOWN:
        new_anchor = [int(x*resize_ratio), int(y*resize_ratio), anchor_type]
        anchors.append(new_anchor)  # 用于画点
        # 绘制当前帧的anchors
        frame_with_anchor = frame.copy()
        drawAnchors(frame_with_anchor, anchors)
        anchor_dict[frame_id] = anchors
        save_anchor_to_file()

# 绘制当前帧的已有锚点
def drawAnchors(img, anchors):
    for _p in anchors:
        p_left_top = (max(_p[0]-anchor_width/2, 0), max(_p[1]-anchor_width/2, 0))
        p_right_down = (min(_p[0]+anchor_width/2, img.shape[1]-1), min(_p[1]+anchor_width/2, img.shape[0]-1))
        cv2.rectangle(img, p_left_top, p_right_down, anchor_color[_p[2] % len(anchor_color)], thickness=4)
        cv2.putText(img, str(_p[2]), p_left_top, cv2.FONT_HERSHEY_SIMPLEX, 3, anchor_color[_p[2] % len(anchor_color)], 2)
    vis_frame = cv2.resize(img, (int(img.shape[1]/resize_ratio), int(img.shape[0]/resize_ratio)))
    cv2.imshow('video', vis_frame)

###################################################
#载入已有标注结果
if os.path.isfile(os.path.join(OUTPUT_DIR, RAW_DATA_FILE+"_anchors_annotation.npy")):
    anchor_dict = np.load(os.path.join(OUTPUT_DIR, RAW_DATA_FILE+"_anchors_annotation.npy")).item()
cv2.namedWindow('video')
cv2.setMouseCallback('video', label_anchor_mouse_callback)
cap = cv2.VideoCapture(RAW_DATA_FILE)  # 读取待标注数据
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

fps=cap.get(cv2.CAP_PROP_FPS)
video_size=(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("raw data fps: {}\nimage size: {}".format(fps, video_size))
tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# 滑动条
def onChange(trackbarValue):
    global frame_id, frame, frame_with_anchor
    cap.set(cv2.CAP_PROP_POS_FRAMES, trackbarValue)
    err, frame = cap.read()
    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos('frame_id','video', frame_id-1)
    # 当前帧的anchors
    anchors = [] if not frame_id in anchor_dict.keys() else anchor_dict[frame_id]    
    # 绘制当前帧的anchors
    frame_with_anchor = frame.copy()
    drawAnchors(frame_with_anchor, anchors)
    vis_frame = cv2.resize(frame_with_anchor, (int(frame_with_anchor.shape[1]/resize_ratio), int(frame_with_anchor.shape[0]/resize_ratio)))
    cv2.imshow('video', vis_frame)
    anchor_type = 0
    pass

frame_id = 0
cv2.createTrackbar( 'frame_id', 'video', 0, tot_frames, onChange)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

while (cap.isOpened()):
    # capture frame-by-frame
    if flag_playing_video or flag_next_frame:
        ret, frame = cap.read()
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos('frame_id','video', frame_id-1)
        flag_next_frame = False
        anchor_type = 0
    if not ret:
        break
    waitkey_time = 1 if flag_playing_video else 0
    # 当前帧的anchors
    anchors = [] if not frame_id in anchor_dict.keys() else anchor_dict[frame_id]    
    # 绘制当前帧的anchors   
    frame_with_anchor = frame.copy()
    drawAnchors(frame_with_anchor, anchors)
    vis_frame = cv2.resize(frame_with_anchor, (int(frame_with_anchor.shape[1]/resize_ratio), int(frame_with_anchor.shape[0]/resize_ratio)))
    cv2.imshow('video', vis_frame)
    input_key = cv2.waitKey(waitkey_time)
    
    if input_key & 0xFF == ord('q'):  # 按q键退出，并保存所有图片
        save_anchor_img_to_file()
        break
    if input_key & 0xFF == 32:  # 按空格播放/暂停
        flag_playing_video = not flag_playing_video
    if input_key & 0xFF in range(ord('0'), ord('9')+1):
        anchor_type = int(input_key & 0xFF) - ord('0')
        print("anchor_type: {}".format(anchor_type))
    if input_key & 0xFF == ord('d'):    # 下一帧
        flag_next_frame = True
        anchor_type = 0
    if input_key & 0xFF == ord('a'):    # 上一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(-1, frame_id-2))
        flag_next_frame = True
        anchor_type = 0
    if input_key & 0xFF == 255:    # delete键
        if len(anchors) > 0:
            del anchors[-1]
            # 绘制当前帧的anchors
            frame_with_anchor = frame.copy()
            drawAnchors(frame_with_anchor, anchors)
            anchor_dict[frame_id] = anchors
            save_anchor_to_file()
    if input_key & 0xFF == ord(','):    # 快速跳转到上一个anchor frame
        _new_frame_id = frame_id
        anchor_frame_rank = anchor_dict.keys()
        anchor_frame_rank.sort()
        for _frame in anchor_frame_rank:
            if (_frame >= frame_id):
                break
            _new_frame_id = _frame
        if _new_frame_id != frame_id:
            onChange(_new_frame_id-1)
    if input_key & 0xFF == ord('.'):    # 快速跳转到下一个anchor frame
        _new_frame_id = frame_id
        anchor_frame_rank = anchor_dict.keys()
        anchor_frame_rank.sort()
        for _frame in anchor_frame_rank:
            if (_frame > frame_id):
                _new_frame_id = _frame
                break
        if _new_frame_id != frame_id:
            onChange(_new_frame_id-1)

# when everything done , release the capture
cap.release()
cv2.destroyAllWindows()