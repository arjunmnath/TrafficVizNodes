from reid_server.config import ServerConfig
from reid_server.global_registry import GlobalRegistry
from reid_server.matcher import Matcher
from reid_server.subscriber import ZMQSubscriber
from reid_server.api import ReIDEventStreamer
from shared.utils import setup_logger
import sys

def run_reid_server(config: ServerConfig):
    logger = setup_logger("ReIDServer")
    logger.info("Starting ReID Server")
    
    registry = GlobalRegistry()
    matcher = Matcher(config, registry)
    subscriber = ZMQSubscriber(bind_address=config.zmq_bind)
    
    api_server = ReIDEventStreamer(port=config.api_port)
    api_server.start()
    
    try:
        while True:
            event = subscriber.receive()
            if event is None:
                continue
                
            global_id = matcher.match(event)
            
            # Format output as requested
            color = event.attributes.color
            attr_str = f"color={color}"
            type_str = event.attributes.type
            if type_str is not None:
                attr_str += f" type={type_str}"
                
            print(f"[GLOBAL_ID={global_id}] camera={event.camera_id} track={event.track_id} class={event.class_label} {attr_str} time={event.timestamp:.2f}")
            sys.stdout.flush()
            
            # Broadcast to SSE API
            api_server.broadcast({
                "global_id": global_id,
                "camera_id": event.camera_id,
                "track_id": event.track_id,
                "class_label": event.class_label,
                "color": color,
                "type": type_str,
                "timestamp": event.timestamp
            })
            
    except KeyboardInterrupt:
        logger.info("Server stopped by user")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zmq_bind", type=str, default="tcp://*:5555")
    parser.add_argument("--api_port", type=int, default=8000)
    args = parser.parse_args()
    
    config = ServerConfig(
        zmq_bind=args.zmq_bind,
        api_port=args.api_port
    )
    run_reid_server(config)
