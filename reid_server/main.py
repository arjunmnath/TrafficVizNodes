from reid_server.config import ServerConfig
from reid_server.global_registry import GlobalRegistry
from reid_server.matcher import Matcher
from reid_server.subscriber import ZMQSubscriber
from reid_server.api import ReIDEventStreamer
from reid_server.event_store import EventStore
from reid_server.crop_extractor import CropExtractor
from reid_server.indexer import RetrievalIndexer
from inference_node.retrieval.encoder import get_retrieval_encoder
from shared.utils import setup_logger
import sys


def run_reid_server(config: ServerConfig):
    logger = setup_logger("ReIDServer")
    logger.info("Starting ReID Server")

    registry = GlobalRegistry()
    matcher = Matcher(config, registry)
    subscriber = ZMQSubscriber(bind_address=config.zmq_bind)

    event_store = None
    try:
        encoder = get_retrieval_encoder(
            model_name=config.retrieval_model,
            device=config.device,
        )
        crop_extractor = CropExtractor(video_sources=config.video_sources)
        indexer = RetrievalIndexer(encoder=encoder, crop_extractor=crop_extractor)
        event_store = EventStore(
            host=config.chroma_host,
            port=config.chroma_port,
            collection_name=config.chroma_collection,
            indexer=indexer,
        )
        logger.info(f"ChromaDB event store initialized with model {config.retrieval_model}")
    except Exception as exc:
        logger.warning(f"ChromaDB unavailable, events will not be persisted: {exc}")

    api_server = ReIDEventStreamer(port=config.api_port)
    api_server.start()

    try:
        while True:
            event = subscriber.receive()
            if event is None:
                continue

            global_id = matcher.match(event)

            color = event.attributes.color
            attr_str = f"color={color}"
            type_str = event.attributes.type
            if type_str is not None:
                attr_str += f" type={type_str}"

            sys.stdout.flush()

            if event_store is not None:
                video_path = config.video_sources.get(event.camera_id)
                event_store.store_event(
                    {
                        "camera_id": event.camera_id,
                        "track_id": event.track_id,
                        "camera_timestamp": event.timestamp,
                        "video_pos_ms": event.video_pos_ms,
                        "bbox": event.bbox,
                        "video_path": video_path,
                    }
                )

            api_server.broadcast(
                {
                    "global_id": global_id,
                    "camera_id": event.camera_id,
                    "track_id": event.track_id,
                    "class_label": event.class_label,
                    "color": color,
                    "type": type_str,
                    "timestamp": event.timestamp,
                }
            )

    except KeyboardInterrupt:
        logger.info("Server stopped by user")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--zmq_bind", type=str, default="tcp://*:5555")
    parser.add_argument("--api_port", type=int, default=8000)
    parser.add_argument("--chroma_host", type=str, default="chromadb")
    parser.add_argument("--chroma_port", type=int, default=8000)
    parser.add_argument("--chroma_collection", type=str, default="track_events")
    parser.add_argument(
        "--retrieval_model",
        type=str,
        default="google/siglip2-base-patch16-224",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--video_sources",
        type=str,
        default="{}",
        help='JSON dict mapping camera_id to video path',
    )
    args = parser.parse_args()

    config = ServerConfig(
        zmq_bind=args.zmq_bind,
        api_port=args.api_port,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
        chroma_collection=args.chroma_collection,
        retrieval_model=args.retrieval_model,
        device=args.device,
        video_sources=json.loads(args.video_sources),
    )
    run_reid_server(config)
