class InferenceConfig:
    """Minimal configuration for model inference."""
    def __init__(
        self,
        backbone: str = "resnet101_ibn_a",
        image_size: tuple = (384, 384),
        pixel_mean: list = (0.485, 0.456, 0.406),
        pixel_std: list = (0.229, 0.224, 0.225),
        device: str = "cuda",
        fp16: bool = True,
        batch_size: int = 128,
        checkpoint_path: str = None,
        neck: str = "bnneck",
        neck_feat: str = "after",
        pooling_method: str = "avg",
        id_loss_type: str = "softmax",
        transformer_type: str = "None",
        stride_size: list = (32, 32),
        camera_embedding: bool = False,
        viewpoint_embedding: bool = False,
        aie_coe: float = 2.5,
        local_f: bool = False,
        drop_path: float = 0.1,
        flip_feats: bool = False,
        cos_layer: bool = False,
    ):
        self.backbone = backbone
        self.image_size = tuple(image_size)
        self.pixel_mean = list(pixel_mean)
        self.pixel_std = list(pixel_std)
        self.device = device
        self.fp16 = fp16
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.neck = neck
        self.neck_feat = neck_feat
        self.pooling_method = pooling_method
        self.id_loss_type = id_loss_type
        self.transformer_type = transformer_type
        self.stride_size = list(stride_size)
        self.camera_embedding = camera_embedding
        self.viewpoint_embedding = viewpoint_embedding
        self.aie_coe = aie_coe
        self.local_f = local_f
        self.drop_path = drop_path
        self.flip_feats = flip_feats
        self.cos_layer = cos_layer

    def to_yacs_mock(self):
        """Converts to a mock object simulating the nested attribute structure expected by make_model."""
        class YacsSubNode:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
            def __repr__(self):
                return str(self.__dict__)
        
        cfg = YacsSubNode()
        cfg.MODEL = YacsSubNode(
            NAME=self.backbone,
            LAST_STRIDE=1,
            PRETRAIN_CHOICE='no',
            PRETRAIN_PATH='',
            FROZEN=-1,
            NECK=self.neck,
            POOLING_METHOD=self.pooling_method,
            ID_LOSS_TYPE=self.id_loss_type,
            Transformer_TYPE=self.transformer_type,
            CAMERA_EMBEDDING=self.camera_embedding,
            VIEWPOINT_EMBEDDING=self.viewpoint_embedding,
            AIE_COE=self.aie_coe,
            LOCAL_F=self.local_f,
            STRIDE_SIZE=self.stride_size,
            DROP_PATH=self.drop_path,
            COS_LAYER=self.cos_layer,
            DEVICE=self.device,
            DEVICE_ID='0',
            DIST_TRAIN=False
        )
        cfg.INPUT = YacsSubNode(
            SIZE_TRAIN=list(self.image_size),
            SIZE_TEST=list(self.image_size),
            PIXEL_MEAN=list(self.pixel_mean),
            PIXEL_STD=list(self.pixel_std)
        )
        cfg.TEST = YacsSubNode(
            NECK_FEAT=self.neck_feat,
            FEAT_NORM='yes',
            FLIP_FEATS='on' if self.flip_feats else 'off',
            WEIGHT=self.checkpoint_path or ""
        )
        cfg.SOLVER = YacsSubNode(
            FP16_ENABLED=self.fp16,
            COSINE_SCALE=64,
            COSINE_MARGIN=0.35
        )
        return cfg

    def __repr__(self):
        return (f"InferenceConfig(backbone={self.backbone}, image_size={self.image_size}, "
                f"device={self.device}, fp16={self.fp16}, batch_size={self.batch_size}, "
                f"checkpoint_path={self.checkpoint_path})")
