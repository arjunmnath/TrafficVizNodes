from pydantic import BaseModel

class CameraConfig(BaseModel):
    camera_id: str = "cam_1"
    video_source: str = "0"  # Can be an integer string for webcam, or path to video file
    zmq_endpoint: str = "tcp://127.0.0.1:5555"
    yolo_model: str = "yolov8s.pt" # Using YOLOv8s for Jetson Nano compatibility
    reid_model: str = "resnet18"   # Using ResNet18 as PyTorch ReID stand-in for edge
    confidence_threshold: float = 0.5
    api_port: int = 8001

