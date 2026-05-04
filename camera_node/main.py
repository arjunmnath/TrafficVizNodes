import cv2
import time
from camera_node.config import CameraConfig
from camera_node.tracker import YOLOTracker
from camera_node.reid import ReIDFeatureExtractor
from camera_node.attributes import AttributeExtractor
from camera_node.publisher import ZMQPublisher
from camera_node.streamer import FrameStreamer
from shared.schemas import TrackEvent, Attributes
from shared.utils import setup_logger

def run_camera_node(config: CameraConfig):
    logger = setup_logger(f"CameraNode-{config.camera_id}")
    logger.info(f"Starting camera node {config.camera_id}")
    
    # Initialize models
    tracker = YOLOTracker(model_path=config.yolo_model, conf=config.confidence_threshold)
    reid = ReIDFeatureExtractor()
    attributes = AttributeExtractor()
    
    # Initialize network
    publisher = ZMQPublisher(endpoint=config.zmq_endpoint)
    
    # Initialize streamer
    streamer = FrameStreamer(port=config.api_port)
    streamer.start()
    
    # Open video source
    source = int(config.video_source) if config.video_source.isdigit() else config.video_source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video source {config.video_source}")
        return
        
    logger.info(f"Started reading from source {config.video_source}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop video for continuous testing
                logger.info("End of video stream, looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            timestamp = time.time()
            
            # 1. Track
            results = tracker.track(frame)
            
            display_frame = frame.copy()
            
            if results and results.boxes and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.int().cpu().numpy()
                cls_ids = results.boxes.cls.int().cpu().numpy()
                
                for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Ensure bbox is within frame
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                        continue
                        
                    # 2. Extract class
                    class_label = "person" if cls_id == 0 else "vehicle"
                    
                    # 3. Extract embedding
                    embedding = reid.extract(crop).tolist()
                    
                    # 4. Extract attributes
                    color = attributes.extract_color(crop)
                    type_str = None
                    if class_label == "vehicle":
                        type_str = attributes.extract_type(int(cls_id))
                        
                    # 5. Create event
                    event = TrackEvent(
                        camera_id=config.camera_id,
                        track_id=int(track_id),
                        timestamp=timestamp,
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        class_label=class_label,
                        embedding=embedding,
                        attributes=Attributes(color=color, type=type_str)
                    )
                    
                    # 6. Publish
                    publisher.publish(event)
                    
                    # 7. Draw on frame
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID:{int(track_id)} {class_label} {color}"
                    cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
            streamer.update_frame(display_frame)
                    
            # Simulate real-time 30 FPS camera feed
            time.sleep(1/30.0)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        logger.info("Camera node stopped")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", type=str, default="cam_1")
    parser.add_argument("--video_source", type=str, default="0")
    parser.add_argument("--zmq_endpoint", type=str, default="tcp://127.0.0.1:5555")
    parser.add_argument("--api_port", type=int, default=8001)
    args = parser.parse_args()
    
    config = CameraConfig(
        camera_id=args.camera_id,
        video_source=args.video_source,
        zmq_endpoint=args.zmq_endpoint,
        api_port=args.api_port
    )
    run_camera_node(config)
