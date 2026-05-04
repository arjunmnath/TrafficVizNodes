import zmq
from shared.schemas import TrackEvent
from shared.utils import setup_logger

class ZMQPublisher:
    def __init__(self, endpoint: str):
        self.logger = setup_logger("ZMQPublisher")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        # We connect to the server's bind endpoint
        self.socket.connect(endpoint)
        self.logger.info(f"Connected publisher to {endpoint}")

    def publish(self, event: TrackEvent):
        try:
            message = event.model_dump_json(by_alias=True)
            self.socket.send_string(message)
        except Exception as e:
            self.logger.error(f"Failed to publish event: {e}")
