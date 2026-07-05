from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich import box

MIN_TIME_GAP_SECONDS = 3.0


def should_keep_detections(
    data: dict,
    min_time_gap_seconds: float,
) -> dict[str, list[dict]]:
    """
    Filter detections so that for each global identity:
      - first occurrence from every camera is kept
      - subsequent occurrences from the same camera are only kept
        if sufficiently far apart in time
    """

    filtered: dict[str, list[dict]] = {}

    for global_id, detections in data.items():
        by_video: dict[str, list[dict]] = defaultdict(list)

        for det in detections:
            by_video[det["video"]].append(det)

        kept: list[dict] = []

        for video_name, video_dets in by_video.items():
            video_dets.sort(
                key=lambda x: (
                    x.get("timestamp_seconds", 0.0),
                    x["frame"],
                )
            )

            last_saved_time: float | None = None

            for det in video_dets:
                current_time = float(det.get("timestamp_seconds", 0.0))

                if last_saved_time is None:
                    kept.append(det)
                    last_saved_time = current_time
                    continue

                if current_time - last_saved_time >= min_time_gap_seconds:
                    kept.append(det)
                    last_saved_time = current_time

        filtered[global_id] = kept

    return filtered


def extract_reid_crops(
    json_path: str,
    video_dir: str,
    output_dir: str,
    min_time_gap_seconds: float = MIN_TIME_GAP_SECONDS,
    headless: bool = False,
) -> None:
    json_path = Path(json_path)
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)

    console = Console()

    # Track statistics
    stats = {
        "Unique IDs": 0,
        "Raw Detections": 0,
        "Filtered Detections": 0,
        "IDs in V1 & V2": 0,
        "Extracted Crops": 0,
        "Missing Videos": 0,
        "Frame Errors": 0,
    }

    # Store last N activity logs
    activity_log: list[str] = []

    def log_event(tag: str, message: str, style: str):
        timestamp = time.strftime("%H:%M:%S")
        log_line = f"[dim][{timestamp}][/dim] [{style}][{tag}][/{style}] {message}"
        if headless:
            console.print(log_line)
        else:
            activity_log.append(log_line)
            if len(activity_log) > 100:
                activity_log.pop(0)

    log_event("SYSTEM", f"Starting extraction pipeline. JSON: [cyan]{json_path}[/cyan]", "cyan")

    if not json_path.exists():
        log_event("ERROR", f"JSON file not found: [bold red]{json_path}[/bold red]", "red")
        if headless:
            sys.exit(1)
        else:
            console.print(f"[bold red]Error: JSON file not found: {json_path}[/bold red]")
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    log_event("SYSTEM", f"Output directory set to: [cyan]{output_dir}[/cyan]", "cyan")

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        log_event("ERROR", f"Failed to load JSON file: {e}", "red")
        if headless:
            sys.exit(1)
        else:
            console.print(f"[bold red]Error: Failed to load JSON: {e}[/bold red]")
            return

    if isinstance(data, list):
        data = {str(item["global_id"]): item.get("occurrences", []) for item in data}

    raw_det_count = sum(len(occs) for occs in data.values())
    stats["Raw Detections"] = raw_det_count
    stats["Unique IDs"] = len(data)
    log_event("SYSTEM", f"Loaded [green]{len(data)}[/green] identities with [green]{raw_det_count}[/green] raw occurrences.", "cyan")

    data = should_keep_detections(
        data,
        min_time_gap_seconds=min_time_gap_seconds,
    )

    filtered_det_count = sum(len(occs) for occs in data.values())
    stats["Filtered Detections"] = filtered_det_count
    log_event("FILTER", f"Filtered occurrences to [green]{filtered_det_count}[/green] (gap threshold: [green]{min_time_gap_seconds}s[/green]).", "green")

    # Count IDs with crops in both V1 and V2
    all_vids = {det["video"] for occurrences in data.values() for det in occurrences}
    all_vids_sorted = sorted(list(all_vids))
    if len(all_vids_sorted) >= 2:
        v1 = all_vids_sorted[0]
        v2 = all_vids_sorted[1]
        both_count = 0
        for occurrences in data.values():
            has_v1 = any(det["video"] == v1 for det in occurrences)
            has_v2 = any(det["video"] == v2 for det in occurrences)
            if has_v1 and has_v2:
                both_count += 1
        stats["IDs in V1 & V2"] = both_count

    by_video: dict[str, list[dict]] = defaultdict(list)

    for global_id, detections in data.items():
        for det in detections:
            det["global_id"] = global_id
            by_video[det["video"]].append(det)

    total_videos = len(by_video)
    log_event("SYSTEM", f"Grouped into [cyan]{total_videos}[/cyan] videos to process.", "cyan")

    # Set up progress bars
    overall_progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30, style="grey37", complete_style="cyan"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    video_progress = Progress(
        TextColumn("[bold yellow]{task.description}"),
        BarColumn(bar_width=30, style="grey37", complete_style="yellow"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )

    videos_task = overall_progress.add_task("Overall Progress", total=total_videos)
    video_task = video_progress.add_task("Active Video Detections", total=0, visible=False)

    config_info = {
        "JSON Path": str(json_path),
        "Video Dir": str(video_dir),
        "Output Dir": str(output_dir),
        "Min Time Gap": f"{min_time_gap_seconds}s",
    }

    current_video_name = "None"
    current_video_status = "Waiting..."

    def generate_layout() -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )

        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=3),
        )

        # Header
        header_text = Text(
            "DMT ReID Crop Extractor & Verifier",
            style="bold white on magenta",
            justify="center",
        )
        layout["header"].update(Panel(header_text, border_style="magenta", box=box.ROUNDED))

        # Left Panel (Settings & Stats)
        left_table = Table.grid(padding=1, expand=True)

        cfg_tbl = Table(title="[bold cyan]Configuration[/bold cyan]", box=box.ROUNDED, expand=True)
        cfg_tbl.add_column("Parameter", style="cyan")
        cfg_tbl.add_column("Value", style="green")
        for k, v in config_info.items():
            cfg_tbl.add_row(k, str(v))

        stats_tbl = Table(title="[bold yellow]Statistics[/bold yellow]", box=box.ROUNDED, expand=True)
        stats_tbl.add_column("Metric", style="yellow")
        stats_tbl.add_column("Count", style="white", justify="right")
        for k, v in stats.items():
            stats_tbl.add_row(k, f"[bold]{v}[/bold]" if v > 0 else str(v))

        left_table.add_row(cfg_tbl)
        left_table.add_row(stats_tbl)

        layout["left"].update(Panel(left_table, border_style="cyan", box=box.ROUNDED))

        # Right Panel (Progress & Events)
        right_layout = Layout()
        right_layout.split_column(
            Layout(name="progress_section", size=9),
            Layout(name="events_section", ratio=1),
        )

        prog_table = Table.grid(padding=1, expand=True)
        prog_table.add_row(
            Text.assemble(("Active Video: ", "bold cyan"), (current_video_name, "bold white"))
        )
        prog_table.add_row(
            Text.assemble(("Status:       ", "bold yellow"), (current_video_status, "white"))
        )
        prog_table.add_row(overall_progress)
        prog_table.add_row(video_progress)

        right_layout["progress_section"].update(
            Panel(
                prog_table,
                title="[bold yellow]Extraction Progress[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )

        # Events (Activity Log)
        events_text = Text()
        for ev in activity_log[-12:]:
            events_text.append(Text.from_markup(ev + "\n"))

        right_layout["events_section"].update(
            Panel(
                events_text,
                title="[bold red]Activity Log[/bold red]",
                border_style="red",
                box=box.ROUNDED,
            )
        )

        layout["body"]["right"].update(right_layout)

        # Footer
        footer_text = Text(
            "Processing crops from multi-camera tracking outputs. Press Ctrl+C to abort.",
            style="italic dim white",
            justify="center",
        )
        layout["footer"].update(Panel(footer_text, border_style="grey37", box=box.ROUNDED))

        return layout

    start_time = time.time()

    def process_loop(live_instance=None):
        nonlocal current_video_name, current_video_status
        for video_idx, (video_name, detections) in enumerate(by_video.items(), 1):
            current_video_name = video_name
            current_video_status = f"Initializing {video_name}"
            if live_instance:
                live_instance.update(generate_layout())

            video_path = video_dir / video_name
            log_event(
                "VIDEO",
                f"Processing [bold]{video_name}[/bold] ({len(detections)} occurrences)",
                "magenta",
            )
            if live_instance:
                live_instance.update(generate_layout())

            if not video_path.exists():
                fallback_path = Path("reid-dmt-backbone") / video_dir / video_name
                if fallback_path.exists():
                    video_path = fallback_path
                else:
                    log_event("WARNING", f"Missing video file: {video_path}", "yellow")
                    stats["Missing Videos"] += 1
                    overall_progress.update(videos_task, advance=1)
                    if live_instance:
                        live_instance.update(generate_layout())
                    continue

            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                log_event("WARNING", f"Could not open video file: {video_path}", "yellow")
                stats["Missing Videos"] += 1
                overall_progress.update(videos_task, advance=1)
                if live_instance:
                    live_instance.update(generate_layout())
                continue

            detections.sort(key=lambda x: x["frame"])

            current_video_status = "Extracting crops..."
            video_progress.update(video_task, total=len(detections), completed=0, visible=True)
            if live_instance:
                live_instance.update(generate_layout())

            current_frame_idx = -1
            frame = None

            for det in detections:
                frame_idx = int(det["frame"])

                if frame_idx != current_frame_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success, frame = cap.read()

                    if not success:
                        log_event(
                            "WARNING",
                            f"Failed reading frame {frame_idx} from {video_name}",
                            "yellow",
                        )
                        stats["Frame Errors"] += 1
                        video_progress.update(video_task, advance=1)
                        if live_instance:
                            live_instance.update(generate_layout())
                        continue

                    current_frame_idx = frame_idx

                if frame is None:
                    video_progress.update(video_task, advance=1)
                    if live_instance:
                        live_instance.update(generate_layout())
                    continue

                x1, y1, x2, y2 = map(int, det["bbox"])
                h, w = frame.shape[:2]

                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))

                if x2 <= x1 or y2 <= y1:
                    video_progress.update(video_task, advance=1)
                    if live_instance:
                        live_instance.update(generate_layout())
                    continue

                crop = frame[y1:y2, x1:x2]

                global_id = str(det["global_id"])
                local_track_id = det.get("local_track_id", -1)

                person_dir = output_dir / global_id
                person_dir.mkdir(parents=True, exist_ok=True)

                stem = Path(video_name).stem
                timestamp = det.get("timestamp_seconds", 0.0)
                similarity = det.get("similarity", 1.0)

                out_name = (
                    f"{stem}"
                    f"_f{frame_idx:06d}"
                    f"_t{local_track_id}"
                    f"_s{timestamp:.2f}"
                    f"_sim{similarity:.4f}.jpg"
                )

                cv2.imwrite(
                    str(person_dir / out_name),
                    crop,
                )

                stats["Extracted Crops"] += 1

                # Avoid log spam for crops, log every 10 crops
                if stats["Extracted Crops"] % 10 == 0 or stats["Extracted Crops"] == filtered_det_count:
                    log_event(
                        "CROP",
                        f"Saved crop for ID {global_id} (Total: {stats['Extracted Crops']})",
                        "blue",
                    )

                video_progress.update(video_task, advance=1)
                if live_instance:
                    live_instance.update(generate_layout())

            cap.release()
            current_video_status = "Finished video"
            video_progress.update(video_task, visible=False)
            overall_progress.update(videos_task, advance=1)
            if live_instance:
                live_instance.update(generate_layout())

    if headless:
        process_loop()
    else:
        with Live(generate_layout(), console=console, refresh_per_second=10) as live:
            process_loop(live)

    elapsed = time.time() - start_time
    log_event(
        "SUCCESS",
        f"Finished extracting [green]{stats['Extracted Crops']}[/green] crops in [cyan]{elapsed:.2f}s[/cyan].",
        "green",
    )

    summary_table = Table(box=box.HEAVY, expand=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Total Processing Time", f"{elapsed:.2f} seconds")
    for k, v in stats.items():
        summary_table.add_row(k, str(v))

    console.print("\n")
    console.print(
        Panel(
            summary_table,
            title="[bold green]CROP EXTRACTION SUMMARY[/bold green]",
            border_style="green",
            expand=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify ReID detections and extract person crops from source videos."
    )
    parser.add_argument(
        "--json_path",
        "-j",
        type=str,
        default="temp.json",
        help="Path to the JSON file containing tracking occurrences.",
    )
    parser.add_argument(
        "--video_dir",
        "-v",
        type=str,
        default="input_vids",
        help="Directory containing the input video files.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="v1",
        help="Directory to save the extracted crops.",
    )
    parser.add_argument(
        "--min_time_gap",
        "-t",
        type=float,
        default=MIN_TIME_GAP_SECONDS,
        help="Minimum time gap in seconds between kept detections of the same identity.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Enable headless mode (disable fullscreen Live TUI, print plain text logs).",
    )

    args = parser.parse_args()

    # Resolve paths relative to script directory if not found in current directory
    script_dir = Path(__file__).resolve().parent

    json_path = Path(args.json_path)
    if not json_path.exists():
        fallback_json = script_dir / args.json_path
        if fallback_json.exists():
            json_path = fallback_json

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        fallback_video_dir = script_dir / args.video_dir
        if fallback_video_dir.exists():
            video_dir = fallback_video_dir

    output_dir = Path(args.output_dir)

    extract_reid_crops(
        json_path=str(json_path),
        video_dir=str(video_dir),
        output_dir=str(output_dir),
        min_time_gap_seconds=args.min_time_gap,
        headless=args.headless,
    )
