from symbol.builder import FasterRcnn as Detector
from symbol.builder import MXNetResNet101V2 as Backbone
from symbol.builder import Neck
from symbol.builder import RpnHead
from symbol.builder import RoiAlign as RoiExtractor
from symbol.builder import BboxC5Head as BboxHead
from mxnext.complicate import normalizer_factory
import numpy as np


def get_config(is_train):
    class General:
        use_groupsoftmax = True
        log_frequency = 20
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_image = 2 if is_train else 1
        fp16 = True

    class KvstoreParam:
        kvstore     = "local"
        batch_image = General.batch_image
        gpus        = [0, 1, 2, 3, 4, 5, 6, 7]
        fp16        = General.fp16


    class NormalizeParam:
        if is_train:
            normalizer = normalizer_factory(type="syncbn", ndev=len(KvstoreParam.gpus))
        else:
            normalizer = normalizer_factory(type="fixbn")


    class BackboneParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer


    class NeckParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer


    class RpnParam:
        fp16 = General.fp16
        normalizer = normalizer_factory(type="fixbn")  # old model does not use BN in RPN head
        batch_image = General.batch_image
        use_groupsoftmax = General.use_groupsoftmax
        num_class = (1 + 2) if use_groupsoftmax else 2

        class anchor_generate:
            scale = (2, 4, 8, 16, 32)
            ratio = (0.5, 1.0, 2.0)
            stride = 16
            image_anchor = 256

        class head:
            conv_channel = 512
            mean = (0, 0, 0, 0)
            std = (1, 1, 1, 1)

        class proposal:
            pre_nms_top_n = 12000 if is_train else 6000
            post_nms_top_n = 2000 if is_train else 1000
            nms_thr = 0.7
            min_bbox_side = 0

        class subsample_proposal:
            proposal_wo_gt = True
            image_roi = 256
            fg_fraction = 0.25
            fg_thr = 0.5
            bg_thr_hi = 0.5
            bg_thr_lo = 0.0

        class bbox_target:
            num_reg_class = 2
            class_agnostic = True
            weight = (1.0, 1.0, 1.0, 1.0)
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (0.1, 0.1, 0.2, 0.2)


    class BboxParam:
        fp16        = General.fp16
        normalizer  = NormalizeParam.normalizer
        num_class   = 1 + 83
        image_roi   = 256
        batch_image = General.batch_image
        use_groupsoftmax = General.use_groupsoftmax

        class regress_target:
            class_agnostic = True
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (0.1, 0.1, 0.2, 0.2)


    class RoiParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        out_size = 7
        stride = 16


    class DatasetParam:
        if is_train:
            image_set = ("coco_train2014", "coco_valminusminival2014", "cctsdb_train")
        else:
            image_set = ("coco_minival2014", )

    backbone = Backbone(BackboneParam)
    neck = Neck(NeckParam)
    rpn_head = RpnHead(RpnParam)
    roi_extractor = RoiExtractor(RoiParam)
    bbox_head = BboxHead(BboxParam)
    detector = Detector()
    if is_train:
        train_sym = detector.get_train_symbol(backbone, neck, rpn_head, roi_extractor, bbox_head)
        rpn_test_sym = None
        test_sym = None
    else:
        train_sym = None
        rpn_test_sym = detector.get_rpn_test_symbol(backbone, neck, rpn_head)
        test_sym = detector.get_test_symbol(backbone, neck, rpn_head, roi_extractor, bbox_head)


    class ModelParam:
        train_symbol = train_sym
        test_symbol = test_sym
        rpn_test_symbol = rpn_test_sym

        from_scratch = False
        random = True
        memonger = False
        memonger_until = "stage3_unit21_plus"

        class pretrain:
            prefix = "pretrain_model/resnet-101"
            epoch = 0
            fixed_param = []


    class OptimizeParam:
        class optimizer:
            type = "sgd"
            lr = 0.01 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
            momentum = 0.9
            wd = 0.0001
            clip_gradient = 5

        class schedule:
            begin_epoch = 0
            end_epoch = 6
            lr_iter = [60000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                       80000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)]

        class warmup:
            type = "gradual"
            lr = 0.0
            iter = 3000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)


    class TestParam:
        min_det_score = 0.05
        max_det_per_image = 100

        process_roidb = lambda x: x
        process_output = lambda x, y: x

        class model:
            prefix = "experiments/{}/checkpoint".format(General.name)
            epoch = OptimizeParam.schedule.end_epoch

        class nms:
            type = "nms"
            thr = 0.5

        class coco:
            annotation = "/ws/data/opendata/coco/annotations/instances_minival2014.json"

    # data processing
    class GroupParam:
        # box 83 classes
        boxv0 = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, \
                              31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, \
                              61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83], dtype=np.float32)
        #COCO benchmark
        boxv1 = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, \
                              31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, \
                              61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 0,  0,  0 ], dtype=np.float32)
        #CCTSDB benchmark
        boxv2 = np.array([0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  \
                              0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  \
                              0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  81, 82, 83], dtype=np.float32)
        
        rpnv0 = np.array([0,  1,  2], dtype=np.float32)     # rpn 3 classes
        rpnv1 = np.array([0,  1,  0], dtype=np.float32)     # COCO benchmark
        rpnv2 = np.array([0,  0,  2], dtype=np.float32)     # CCTSDB benchmark

        rpn_groups = [rpnv0, rpnv1, rpnv2]
        box_groups = [boxv0, boxv1, boxv2]


    class ResizeParam:
        short = 800
        long = 1200 if is_train else 2000


    class PadParam:
        short = 800
        long = 1200
        max_num_gt = 100


    class AnchorTarget2DParam:
        class generate:
            short = 800 // 16
            long = 1200 // 16
            stride = 16
            scales = (2, 4, 8, 16, 32)
            aspects = (0.5, 1.0, 2.0)
            use_groupsoftmax = General.use_groupsoftmax

        class assign:
            allowed_border = 0
            pos_thr = 0.7
            neg_thr = 0.3
            min_pos_thr = 0.0

        class sample:
            image_anchor = 256
            pos_fraction = 0.5

        def gtclass2rpn(gtclass):
            class_gap = 80
            gtclass[gtclass > class_gap] = -1
            gtclass[gtclass > 0] = 1
            gtclass[gtclass < 0] = 2
            return gtclass


    class RenameParam:
        mapping = dict(image="data")


    from core.detection_input import ReadRoiRecord, Resize2DImageBbox, \
        ConvertImageFromHwcToChw, Flip2DImageBbox, Pad2DImageBbox, \
        RenameRecord, AnchorTarget2D, GroupRead

    if is_train:
        transform = [
            ReadRoiRecord(None),
            Resize2DImageBbox(ResizeParam),
            Flip2DImageBbox(),
            Pad2DImageBbox(PadParam),
            ConvertImageFromHwcToChw(),
            AnchorTarget2D(AnchorTarget2DParam),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data", "im_info", "gt_bbox"]
        label_name = ["rpn_cls_label", "rpn_reg_target", "rpn_reg_weight"]
        if General.use_groupsoftmax:
            data_name.append("rpn_group")
            data_name.append("box_group")
            transform.append(GroupRead(GroupParam))
    else:
        transform = [
            ReadRoiRecord(None),
            Resize2DImageBbox(ResizeParam),
            ConvertImageFromHwcToChw(),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data", "im_info", "im_id", "rec_id"]
        label_name = []

    import core.detection_metric as metric

    rpn_acc_metric = metric.AccWithIgnore(
        "RpnAcc",
        ["rpn_cls_loss_output"],
        ["rpn_cls_label"]
    )
    rpn_l1_metric = metric.L1(
        "RpnL1",
        ["rpn_reg_loss_output"],
        ["rpn_cls_label"]
    )
    # for bbox, the label is generated in network so it is an output
    box_acc_metric = metric.AccWithIgnore(
        "RcnnAcc",
        ["bbox_cls_loss_output", "bbox_label_blockgrad_output"],
        []
    )
    box_l1_metric = metric.L1(
        "RcnnL1",
        ["bbox_reg_loss_output", "bbox_label_blockgrad_output"],
        []
    )

    metric_list = [rpn_acc_metric, rpn_l1_metric, box_acc_metric, box_l1_metric]

    return General, KvstoreParam, RpnParam, RoiParam, BboxParam, DatasetParam, \
           ModelParam, OptimizeParam, TestParam, \
           transform, data_name, label_name, metric_list
