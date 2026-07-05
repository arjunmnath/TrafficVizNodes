from pydantic import BaseModel


class CameraConfig(BaseModel):
    camera_id: str = "cam_1"
    video_source: str = "0"  # Can be an integer string for webcam, or path to video file
    zmq_endpoint: str = "tcp://127.0.0.1:5555"
    yolo_model: str = "yolov8s.pt"  # Using YOLOv8s for Jetson Nano compatibility
    reid_model_name: str = (
        "resnet101_ibn_a"  # DMT backbone: resnet101_ibn_a, resnext101_ibn_a, etc.
    )
    reid_model_path: str = ""  # Path to .pth trained checkpoint
    reid_flip_augment: bool = False  # Horizontal flip TTA (2x inference cost)
    confidence_threshold: float = 0.5
    api_port: int = 8001
