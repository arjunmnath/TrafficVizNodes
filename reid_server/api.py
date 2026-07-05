from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import json
import asyncio
from shared.utils import setup_logger


class ReIDEventStreamer:
    def __init__(self, port: int):
        self.port = port
        self.logger = setup_logger("ReIDEventStreamer")
        self.app = FastAPI()
        self.queues = []
        self.lock = threading.Lock()

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/events")
        async def sse_events(request: Request):
            my_queue = []
            with self.lock:
                self.queues.append(my_queue)

            async def event_generator():
                try:
                    while True:
                        if await request.is_disconnected():
                            break

                        batch = []
                        with self.lock:
                            if my_queue:
                                batch = list(my_queue)
                                my_queue.clear()

                        for item in batch:
                            yield f"data: {json.dumps(item)}\n\n"

                        if not batch:
                            await asyncio.sleep(0.1)
                finally:
                    with self.lock:
                        if my_queue in self.queues:
                            self.queues.remove(my_queue)

            return StreamingResponse(event_generator(), media_type="text/event-stream")

    def broadcast(self, event_data: dict):
        with self.lock:
            for q in self.queues:
                q.append(event_data)

    def start(self):
        def _run():
            uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="error")

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()
        self.logger.info(f"Started ReID SSE streamer on port {self.port}")
