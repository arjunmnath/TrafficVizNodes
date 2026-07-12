class InferenceConfig:
    """Minimal configuration for model inference."""

    def __init__(
        self,
        backbone: str,
        image_size: tuple = (256, 256),
        device: str = "cuda",
        device_id: str = "0",
        fp16: bool = True,
        batch_size: int = 128,
        checkpoint_path: str = None,
        flip_feats: bool = True,
        last_stride: int = 1,
        neck: str = "bnneck",
        frozen: int = -1,
        pooling_method: str = "avg",
        re_ranking: bool = False,
        re_ranking_track: bool = False,
        neck_feat: str = "after",
        feat_norm: str = "yes",
        dist_mat: str = "dist_mat.npy",
        output_dir: str = "",
    ):
        self.backbone = backbone
        self.image_size = tuple(image_size)
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]
        self.device = device
        self.device_id = device_id
        self.fp16 = fp16
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.flip_feats = flip_feats
        self.last_stride = last_stride
        self.neck = neck
        self.frozen = frozen
        self.pooling_method = pooling_method
        self.re_ranking = re_ranking
        self.re_ranking_track = re_ranking_track
        self.neck_feat = neck_feat
        self.feat_norm = feat_norm
        self.dist_mat = dist_mat
        self.output_dir = output_dir

    def to_yacs_mock(self):
        """Converts to a mock object simulating the nested attribute structure expected by make_model."""

        class YacsSubNode:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        cfg = YacsSubNode()
        cfg.MODEL = YacsSubNode(
            NAME=self.backbone,
            LAST_STRIDE=self.last_stride,
            PRETRAIN_CHOICE="no",
            PRETRAIN_PATH="",
            FROZEN=self.frozen,
            NECK=self.neck,
            POOLING_METHOD=self.pooling_method,
            ID_LOSS_TYPE="softmax",
            Transformer_TYPE="None",
            CAMERA_EMBEDDING=False,
            VIEWPOINT_EMBEDDING=False,
            AIE_COE=2.5,
            LOCAL_F=False,
            STRIDE_SIZE=(32, 32),
            DROP_PATH=0.1,
            COS_LAYER=False,
            DEVICE=self.device,
            DEVICE_ID=self.device_id,
            DIST_TRAIN=False,
        )
        cfg.INPUT = YacsSubNode(
            SIZE_TRAIN=list(self.image_size),
            SIZE_TEST=list(self.image_size),
            PIXEL_MEAN=self.pixel_mean,
            PIXEL_STD=self.pixel_std,
        )
        cfg.TEST = YacsSubNode(
            IMS_PER_BATCH=self.batch_size,
            RE_RANKING=self.re_ranking,
            RE_RANKING_TRACK=self.re_ranking_track,
            NECK_FEAT=self.neck_feat,
            FEAT_NORM=self.feat_norm,
            DIST_MAT=self.dist_mat,
            FLIP_FEATS="on" if self.flip_feats else "off",
            WEIGHT=self.checkpoint_path or "",
        )
        cfg.SOLVER = YacsSubNode(FP16_ENABLED=self.fp16, COSINE_SCALE=64, COSINE_MARGIN=0.35)
        cfg.OUTPUT_DIR = self.output_dir
        return cfg

    def __repr__(self):
        return (
            f"InferenceConfig(backbone={self.backbone}, image_size={self.image_size}, "
            f"device={self.device}, fp16={self.fp16}, batch_size={self.batch_size}, "
            f"checkpoint_path={self.checkpoint_path})"
        )


class EnsembleConfig:
    """Configuration for the ensemble ReID model."""

    def __init__(
        self,
        backbones: list = None,
        checkpoint_paths: list = None,
        image_size: tuple = (256, 256),
        device: str = "cuda",
        fp16: bool = True,
        flip_feats: bool = True,
        batch_size: int = 128,
    ):
        self.backbones = backbones or [
            "resnet101_ibn_a",
            "resnet101_ibn_a",
            "resnext101_ibn_a",
        ]
        self.checkpoint_paths = checkpoint_paths or []
        self.image_size = tuple(image_size)
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]
        self.device = device
        self.fp16 = fp16
        self.flip_feats = flip_feats
        self.batch_size = batch_size

    def __repr__(self):
        return (
            f"EnsembleConfig(backbones={self.backbones}, checkpoint_paths={self.checkpoint_paths}, "
            f"image_size={self.image_size}, device={self.device}, fp16={self.fp16}, "
            f"flip_feats={self.flip_feats}, batch_size={self.batch_size})"
        )
