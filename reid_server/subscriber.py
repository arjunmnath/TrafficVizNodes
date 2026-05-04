import zmq
from shared.schemas import TrackEvent
from shared.utils import setup_logger

class ZMQSubscriber:
    def __init__(self, bind_address: str):
        self.logger = setup_logger("ZMQSubscriber")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(bind_address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "") # Subscribe to all
        self.logger.info(f"Subscriber bound to {bind_address}")

    def receive(self) -> TrackEvent:
        try:
            message = self.socket.recv_string()
            return TrackEvent.model_validate_json(message)
        except Exception as e:
            self.logger.error(f"Failed to receive/parse event: {e}")
            return None
