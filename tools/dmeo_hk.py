import glob
from tools.test import *
import json

import cv2
import time
import sys
sys.path.append("/home/dabai/project/HKcam/hkcam/HKcam/build")
#sys.path.append("/home/dabai/project/HKcam/hkcam")
import HKcam
def proccess_loss(cfg):
    if 'reg' not in cfg:
        cfg['reg'] = {'loss': 'L1Loss'}
    else:
        if 'loss' not in cfg['reg']:
            cfg['reg']['loss'] = 'L1Loss'
    if 'cls' not in cfg:
        cfg['cls'] = {'split': True}
    cfg['weight'] = cfg.get('weight', [1, 1, 36])  # cls, reg, mask

def add_default(conf, default):
    default.update(conf)
    return default
def load_config(file):
    config = json.load(open(file))
    # deal with network
    if 'network' not in config:
        print('Warning: network lost in config. This will be error in next version')
        config['network'] = {}
    # deal with loss
    if 'loss' not in config:
        config['loss'] = {}
    proccess_loss(config['loss'])
    # deal with lr
    if 'lr' not in config:
        config['lr'] = {}
    default = {
        'feature_lr_mult': 1.0,
        'rpn_lr_mult': 1.0,
        'mask_lr_mult': 1.0,
        'type': 'log',
        'start_lr': 0.03
    }
    default.update(config['lr'])
    config['lr'] = default
    return config

def track_init(img, x, y, w, h, device):
    cfg = load_config('config_davis.json')
    # Setup Model
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    siammask = load_pretrain(siammask, 'SiamMask_VOT.pth')
    siammask.eval().to(device)
    target_pos = np.array([x + w / 2, y + h / 2])
    target_sz = np.array([w, h])
    state = siamese_init(img, target_pos, target_sz, siammask, cfg['hp'], device=device)
    return state


def tracking(img, state, device):
    state = siamese_track(state, img, mask_enable=True, refine_enable=True, device=device)  # track
    location = state['ploygon'].flatten()
    mask = state['mask'] > state['p'].seg_thr
    return state, location, mask


if __name__ == '__main__':
    #userId = HKcam.init("192.168.100.250", "admin", "dabai521")   #110 102
    userId = HKcam.init("192.168.100.111", "admin", "dabai521")
    greeen = (0, 255, 0)
    initX = 0
    initY = 0
    initP = 0
    initT = 0
    moveX = 0
    moveY = 0
    moveP = 0
    moveT = 0
    moveDirection = 0
    startFlag = True
    moveFlag = False
    startTime = 0
    half_time = 0
    time_move = 0
    time_after_move = 10000  #防止跟丢时相机一直转
    time_flag = True
    print(userId)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    cap = cv2.VideoCapture(0)#'rtsp://admin:dabai521@192.168.100.111:554/Streaming/Channels/001'
    ret, frame_hk = cap.read()
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', frame_hk, False, False)
        x, y, w, h = init_rect
    except:
        exit()
    initX = x
    initY = y
    state = track_init(frame_hk, x, y, w, h, device)
    width = frame_hk.shape[1]
    while True:
        ret, frame_ = cap.read()
        ret, frame_ = cap.read()
        ret, frame_ = cap.read()
        ret, frame_ = cap.read()
        if not ret:
            continue
        state, location, mask = tracking(frame_, state, device)
        initPTZList = HKcam.get_cam_PTZ(userId)
        if time_flag:
            startTime = time.time()
            time_flag = False
        if startFlag == True:
            initP = initPTZList[0][0]
            initT = initPTZList[0][1]
            startFlag = False
        moveX = (location[0]+location[2]+location[4]+location[6])/4.0
        moveY = (location[1]+location[3]+location[5]+location[7])/4.0
        if moveX - initX > 60:
            if moveFlag == False:
                time_after_move = time.time()
            moveDirection = 24#right
            moveFlag = True
        elif moveX - initX < -60:
            if moveFlag == False:
                time_after_move = time.time()
            moveDirection = 23
            moveFlag = True


        if moveFlag == True:
            time_move = time.time()
            PTZList = HKcam.get_cam_PTZ(userId)
            moveP = PTZList[0][0]
            moveT = PTZList[0][1]
            HKcam.start_move(moveDirection, 4)  #userId,


            print('continue moveing_time---->>', time_move - time_after_move)
            if abs(time_move - time_after_move) > 1.5 and time_after_move != 10000:  # 连续转动时间超过2s停止,防止跟丢时相机一直转
                HKcam.stop_move(23, 0)
                moveFlag = False
                print('tracking false!!!!!!!!!')
                time_after_move = time.time()
                time_move = time.time()


            moveX_temp = 0
            print('coordinate----------',moveP,initP)
            ret_, frame_ = cap.read()
            ret_, frame_ = cap.read()
            ret_, frame_ = cap.read()
            ret_, frame_ = cap.read()
            sta, location_, mas = tracking(frame_, state, device)
            moveX_temp = (location_[0] + location_[2] + location_[4] + location_[6]) / 4
            if moveX_temp < width/2:
                moveDirection = 23
            elif moveX_temp > width/2:
                moveDirection = 24

            if moveX_temp > width/2 -150 and moveX_temp < width/2 + 150:
                HKcam.stop_move(23, 0)
                moveFlag = False
            ret_, frame_ = cap.read()
            ret_, frame_ = cap.read()
            ret_, frame_ = cap.read()
            ret_, frame_ = cap.read()
            sta, location_, mas = tracking(frame_, state, device)
            moveX_end = (location_[0] + location_[2] + location_[4] + location_[6]) / 4
            initX = moveX_end
            #print('continue moveing_time---->>', time_move - time_after_move)
        if abs(abs(moveP) - abs(initP)) >60: #145
            dtime = time.time() - startTime
            if moveP>initP and initP != 0 and dtime>1.5:
                print('go_time:',dtime,'s')
                half_time = dtime
            elif moveP<initP and initP != 0  and dtime>1.5:
                print('back_time:',dtime,'s')
                print('all_time:',dtime+half_time,'s')
            initP = moveP
            time_flag = True



        '''if abs(time_move - time_after_move) > 2 and time_after_move != 10000:  # 连续转动时间超过2s停止,防止跟丢时相机一直转
            HKcam.stop_move(23, 0)
            moveFlag = False
            print('tracking false!!!!!!!!!')
            time_after_move = time.time()
            time_move = time.time()'''



        frame_[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame_[:, :, 2]
        cv2.polylines(frame_, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        cv2.imshow('SiamMask', frame_)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

