import argparse
import subprocess
import time
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Run MTMC Tracking and ReID System")
    parser.add_argument("--videos", nargs="+", required=True, help="List of video files (or camera indices) to simulate cameras")
    parser.add_argument("--zmq_port", type=int, default=5555, help="ZMQ port for pub/sub")
    parser.add_argument("--reid_model_name", type=str, default="resnet101_ibn_a",
                        help="DMT backbone name (resnet101_ibn_a, resnext101_ibn_a, etc.)")
    parser.add_argument("--reid_model_path", type=str, default="",
                        help="Path to trained .pth checkpoint")
    parser.add_argument("--reid_flip_augment", action="store_true",
                        help="Enable horizontal flip TTA")
    args = parser.parse_args()

    videos = args.videos
    num_cameras = len(videos)
    
    print(f"Starting MTMC System with {num_cameras} cameras.")
    
    processes = []
    
    # Ensure PYTHONPATH is set to project root
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.dirname(__file__))
    
    # 1. Start ReID Server
    server_cmd = [
        sys.executable, "-m", "reid_server.main",
        "--zmq_bind", f"tcp://*:{args.zmq_port}",
        "--api_port", "8000"
    ]
    print(f"Starting ReID Server: {' '.join(server_cmd)}")
    server_proc = subprocess.Popen(server_cmd, env=env)
    processes.append(server_proc)
    
    # Give server a moment to bind the socket
    time.sleep(2)
    
    # 2. Start Camera Nodes
    for idx, video in enumerate(videos):
        camera_id = f"cam_{idx+1}"
        api_port = str(8001 + idx)
        cam_cmd = [
            sys.executable, "-m", "camera_node.main",
            "--camera_id", camera_id,
            "--video_source", video,
            "--zmq_endpoint", f"tcp://127.0.0.1:{args.zmq_port}",
            "--api_port", api_port,
            "--reid_model_name", args.reid_model_name,
        ]
        if args.reid_model_path:
            cam_cmd.extend(["--reid_model_path", args.reid_model_path])
        if args.reid_flip_augment:
            cam_cmd.append("--reid_flip_augment")
        print(f"Starting Camera Node {camera_id}: {' '.join(cam_cmd)}")
        cam_proc = subprocess.Popen(cam_cmd, env=env)
        processes.append(cam_proc)
        
    print("All processes started. Press Ctrl+C to stop.")
    
    try:
        # Wait for all processes
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all processes...")
        for p in processes:
            p.terminate()
            p.wait()
        print("System stopped.")

if __name__ == "__main__":
    main()
