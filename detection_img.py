import os

from core.detection_module import DetModule
from core.detection_input import Loader
from utils.load_model import load_checkpoint
from operator_py.nms import py_nms_wrapper
from utils import callback
from mxnet.base import _as_list

from six.moves import reduce
from six.moves.queue import Queue
from threading import Thread
import argparse
import importlib
import mxnet as mx
import numpy as np
import six.moves.cPickle as pkl
import time
import json
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Test Detection')
    # general
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--gpu_id', help='gpu_id', type=int, default=0)
    parser.add_argument('--epoch', help='load params epoch', type=int, default=0)
    parser.add_argument('--thr', help='detection threshold', type=float, default=0.80)
    parser.add_argument('--path', help='images path to detect', type=str)
    args = parser.parse_args()

    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    return args, config

if __name__ == "__main__":
    # os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    args, config = parse_args()

    pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
    transform, data_name, label_name, metric_list = config.get_config(is_train=False)

    nms = py_nms_wrapper(pTest.nms.thr)
    sym = pModel.test_symbol
    pshort = 800
    plong  = 2000

    arg_params, aux_params = load_checkpoint(pTest.model.prefix, args.epoch)
    mod = DetModule(sym, data_names=["data", "im_info", "im_id", "rec_id"], context=mx.gpu(args.gpu_id))
    provide_data = [("data", (1, 3, pshort, plong)), ("im_info", (1, 3)), ("im_id", (1,)), ("rec_id", (1,))]
    mod.bind(data_shapes=provide_data, for_training=False)
    mod.set_params(arg_params, aux_params, allow_extra=False)

    image_list = []
    if os.path.isfile(args.path):
        if ".txt" in args.path:
            list_file = open(args.path, 'r')
            list_lines = list_file.readlines()
            list_file.close()
            (fpath, fname) = os.path.split(args.path)
            for aline in list_lines:
                uints = aline.split(' ')
                imgpath = os.path.join(fpath, uints[0])
                image_list.append(imgpath)
        else:
            image_list.append(args.path)
    else:
        for fname in os.listdir(args.path):
            fpath = os.path.join(args.path, fname)
            if os.path.isfile(fpath):
                image_list.append(fpath)
    
    for imgpath in image_list:
        img   = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        image = img[:, :, ::-1]
        short = image.shape[0]
        long  = image.shape[1]
        scale = min(pshort / short, plong / long)
        image = cv2.resize(image, None, None, scale, scale, interpolation=cv2.INTER_LINEAR)
        # exactly as opencv
        h, w = image.shape[:2]
        im_info = (h, w, scale)
        # shape = (plong, pshort, 3) if h >= w else (pshort, plong, 3)
        shape = (pshort, plong, 3)
        padded_image = np.zeros(shape, dtype=np.float32)
        padded_image[:h, :w] = image
        padded_image = padded_image.transpose((2, 0, 1))
        img_array = []
        img_array.append(padded_image)
        iminfo_array = []
        iminfo_array.append(im_info)
        im_id = mx.nd.array([1])
        rec_id = mx.nd.array([1])
        data = [mx.nd.array(img_array)]
        data.append(mx.nd.array(iminfo_array))
        data.append(im_id)
        data.append(rec_id)
        mbatch = mx.io.DataBatch(data=data, provide_data=provide_data)
    
        start_t = time.time()
        mod.forward(mbatch, is_train=False)
        outs = [x.asnumpy() for x in mod.get_outputs()]
        im_info   = outs[2]       # h_raw, w_raw, scale
        cls_score = outs[3]
        bbox_xyxy = outs[4]
        if cls_score.ndim == 3:
            cls_score = cls_score[0]
            bbox_xyxy = bbox_xyxy[0]
        bbox_xyxy = bbox_xyxy / scale       # scale to original image scale
        cls_score = cls_score[:, 1:]        # remove background score
        # TODO: the output shape of class_agnostic box is [n, 4], while class_aware box is [n, 4 * (1 + class)]
        bbox_xyxy = bbox_xyxy[:, 4:] if bbox_xyxy.shape[1] != 4 else bbox_xyxy

        final_dets = {}
        for cid in range(cls_score.shape[1]):
            score = cls_score[:, cid]
            if bbox_xyxy.shape[1] != 4:
                cls_box = bbox_xyxy[:, cid * 4:(cid + 1) * 4]
            else:
                cls_box = bbox_xyxy
            valid_inds = np.where(score > args.thr)[0]
            box   = cls_box[valid_inds]
            score = score[valid_inds]
            det = np.concatenate((box, score.reshape(-1, 1)), axis=1).astype(np.float32)
            final_dets[cid] = nms(det)
        end_t = time.time()
        print("detection use: %.3f seconds." % (end_t - start_t))

        for cid in final_dets:
            det = final_dets[cid]
            if det.shape[0] == 0:
                continue
            scores = det[:, -1]
            x1 = det[:, 0]
            y1 = det[:, 1]
            x2 = det[:, 2]
            y2 = det[:, 3]
            for k in range(det.shape[0]):
                bbox  = [float(x1[k]), float(y1[k]), float(x2[k]), float(y2[k])]
                score =  float(scores[k])
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                # cv2.putText(img, "{}:{:.2}".format(str(cid), score), (int(bbox[0]), int(bbox[1] - 10)), 4, 0.6, (0, 0, 255))
        (filepath, filename) = os.path.split(imgpath)
        cv2.imwrite(filename, img)
    exit()


