import cv2
import threading
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from shared.utils import setup_logger


class FrameStreamer:
    def __init__(self, port: int):
        self.port = port
        self.logger = setup_logger(f"FrameStreamer-{port}")
        self.latest_frame = None
        self.lock = threading.Lock()

        self.app = FastAPI()

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/mjpeg")
        async def mjpeg_stream():
            return StreamingResponse(
                self._generate(), media_type="multipart/x-mixed-replace; boundary=frame"
            )

    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()

    async def _generate(self):
        while True:
            with self.lock:
                frame = self.latest_frame

            if frame is not None:
                ret, buffer = cv2.imencode(".jpg", frame)
                if ret:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                    )

            await asyncio.sleep(0.05)

    def start(self):
        def _run():
            uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="error")

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()
        self.logger.info(f"Started MJPEG streamer on port {self.port}")
