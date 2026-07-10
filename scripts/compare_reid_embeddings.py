#!/usr/bin/env python3
"""
Cosine similarity analysis script to compare feature embeddings of occurrences between tracked global IDs and output statistics and heatmaps.
Reads the ReID pipeline JSON tracker output, computes pairwise cosine similarities, and saves a heatmap plot.
Supports comparing a single global ID, two global IDs, or multiple global IDs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def load_occurrences_by_id(json_path: Path) -> dict[str, list[dict]]:
    """Loads and parses the tracker output JSON file, indexing occurrences by global_id."""
    if not json_path.exists():
        console.print(f"[bold red]Error: JSON file not found at {json_path}[/bold red]")
        sys.exit(1)

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[bold red]Error: Failed to read or parse JSON file: {e}[/bold red]")
        sys.exit(1)

    # Normalize format to: dict mapping global ID (str) to list of occurrences
    by_id = {}
    if isinstance(data, list):
        for item in data:
            if "global_id" in item:
                gid = str(item["global_id"])
                by_id[gid] = item.get("occurrences", [])
    elif isinstance(data, dict):
        by_id = {str(k): v for k, v in data.items()}
    else:
        console.print("[bold red]Error: Unsupported JSON format. Expected list or dict.[/bold red]")
        sys.exit(1)

    return by_id


def extract_embeddings_and_labels(
    occs: list[dict], global_id: str, embeddings_dict: dict = None
) -> tuple[np.ndarray, list[str]]:
    """Extracts embeddings and creates human-readable labels for a list of occurrences."""
    labels = []
    for occ in occs:
        video = occ.get("video", "unknown")
        frame = occ.get("frame", 0)
        timestamp = occ.get("timestamp_seconds", 0.0)
        labels.append(f"ID {global_id} | {video} | F{frame} ({timestamp:.1f}s)")

    embeddings = None
    if embeddings_dict is not None and global_id in embeddings_dict:
        embeddings = embeddings_dict[global_id]
    else:
        # Fallback to checking occurrence dictionary if stored there (backward compatibility)
        embeddings_list = []
        for occ in occs:
            emb = occ.get("embedding", None)
            if emb is not None:
                embeddings_list.append(emb)
        if embeddings_list:
            embeddings = np.array(embeddings_list, dtype=np.float32)

    if embeddings is None or len(embeddings) == 0:
        return np.empty((0, 0)), []

    return embeddings, labels


def compute_pairwise_cosine_similarity(all_embeddings: np.ndarray) -> np.ndarray:
    """
    Computes the full pairwise cosine similarity matrix of a set of embeddings.
    """
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    norm_embeddings = all_embeddings / norms
    return np.dot(norm_embeddings, norm_embeddings.T)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare embeddings of occurrences for one or more global IDs and plot similarities."
    )
    parser.add_argument(
        "--json_path",
        "-j",
        type=str,
        required=True,
        help="Path to ReID pipeline tracker JSON output file.",
    )
    parser.add_argument(
        "--ids",
        type=str,
        nargs="+",
        help="List of global IDs to compare (e.g. --ids 1 or --ids 1 2 or --ids 1 2 3).",
    )
    parser.add_argument(
        "--id1",
        type=str,
        help="First global ID to compare (deprecated, use --ids instead).",
    )
    parser.add_argument(
        "--id2",
        type=str,
        help="Second global ID to compare (deprecated, use --ids instead).",
    )
    parser.add_argument(
        "--output_plot",
        "-o",
        type=str,
        default="similarity_matrix.png",
        help="Output filepath for the similarity matrix heatmap visualization.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="coolwarm",
        help="Matplotlib colormap to use (e.g. coolwarm, viridis, plasma, RdYlBu).",
    )
    parser.add_argument(
        "--show",
        "-s",
        action="store_true",
        help="Show the interactive plot window before saving.",
    )
    args = parser.parse_args()

    # Determine targeted IDs
    target_ids = []
    if args.ids:
        target_ids = [str(x) for x in args.ids]
    else:
        if args.id1:
            target_ids.append(str(args.id1))
        if args.id2:
            target_ids.append(str(args.id2))

    if not target_ids:
        console.print("[bold red]Error: No global IDs specified. Please use --ids, or --id1/--id2.[/bold red]")
        sys.exit(1)

    # Resolve input JSON path
    json_path = Path(args.json_path)
    if not json_path.exists():
        # Fallback to workspace root folder if path was relative
        script_dir = Path(__file__).resolve().parent
        workspace_root = script_dir.parent
        fallback_path = workspace_root / args.json_path
        if fallback_path.exists():
            json_path = fallback_path

    console.print(f"[bold cyan]Loading data from:[/bold cyan] {json_path}")
    by_id = load_occurrences_by_id(json_path)

    # Resolve and load NPZ file if it exists
    embeddings_dict = {}
    npz_path = json_path.with_suffix(".npz")
    if npz_path.exists():
        try:
            embeddings_dict = np.load(npz_path)
        except Exception as e:
            console.print(f"[bold red]Warning: Failed to load embeddings NPZ file: {e}[/bold red]")

    # Validate all IDs exist in dataset
    available_ids = sorted(list(by_id.keys()), key=lambda x: int(x) if x.isdigit() else x)
    for gid in target_ids:
        if gid not in by_id:
            console.print(f"[bold red]Error: Global ID '{gid}' not found in the dataset.[/bold red]")
            console.print(f"Available IDs: {available_ids}")
            sys.exit(1)

    # Load and process embeddings per ID
    id_occs = {}
    all_embeddings_list = []
    all_labels = []
    id_indices = {}  # Map gid -> (start_idx, end_idx)

    current_idx = 0
    for gid in target_ids:
        occs = by_id[gid]
        # Sort chronologically (by video, then frame number)
        occs = sorted(occs, key=lambda x: (x.get("video", ""), x.get("frame", 0)))
        id_occs[gid] = occs

        embs, labels = extract_embeddings_and_labels(occs, gid, embeddings_dict)
        n = len(embs)
        console.print(f"[bold green]Global ID {gid}:[/bold green] Found {n} occurrences with embeddings.")

        if n == 0:
            console.print(f"[bold red]Error: Global ID '{gid}' has zero occurrences with embeddings.[/bold red]")
            sys.exit(1)

        all_embeddings_list.append(embs)
        all_labels.extend(labels)
        id_indices[gid] = (current_idx, current_idx + n)
        current_idx += n

    total_occs = len(all_labels)
    all_embeddings = np.concatenate(all_embeddings_list, axis=0)

    # Compute similarity matrix
    similarity_matrix = compute_pairwise_cosine_similarity(all_embeddings)

    # Helper to calculate stats excluding diagonal for self-similarities
    def get_stats(matrix: np.ndarray, exclude_diag: bool = False) -> tuple[float, float, float, float]:
        if matrix.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        if exclude_diag and matrix.shape[0] == matrix.shape[1]:
            n = matrix.shape[0]
            if n <= 1:
                return 1.0, 0.0, 1.0, 1.0
            vals = matrix[~np.eye(n, dtype=bool)]
        else:
            vals = matrix.flatten()
        return float(np.mean(vals)), float(np.std(vals)), float(np.min(vals)), float(np.max(vals))

    # Print results to the console in a nice Table
    stats_table = Table(title="Embedding Similarity Statistics", box=box.ROUNDED)
    stats_table.add_column("Comparison Group", style="cyan", justify="left")
    stats_table.add_column("Count", style="white", justify="right")
    stats_table.add_column("Mean Similarity", style="green", justify="right")
    stats_table.add_column("Std Dev", style="yellow", justify="right")
    stats_table.add_column("Min Similarity", style="red", justify="right")
    stats_table.add_column("Max Similarity", style="green", justify="right")

    # Add Intra stats for each ID
    for gid in target_ids:
        start, end = id_indices[gid]
        n = end - start
        sub_matrix = similarity_matrix[start:end, start:end]
        mean_val, std_val, min_val, max_val = get_stats(sub_matrix, exclude_diag=True)
        stats_table.add_row(
            f"ID {gid} vs ID {gid} (Intra)",
            f"{n}x{n} ({n * (n - 1) if n > 1 else 1} pairs)",
            f"{mean_val:.4f}",
            f"{std_val:.4f}",
            f"{min_val:.4f}",
            f"{max_val:.4f}",
        )

    # Add Inter stats if comparing 2 or more IDs
    if len(target_ids) >= 2:
        stats_table.add_section()
        for i in range(len(target_ids)):
            for j in range(i + 1, len(target_ids)):
                gid1 = target_ids[i]
                gid2 = target_ids[j]
                start1, end1 = id_indices[gid1]
                start2, end2 = id_indices[gid2]
                n1 = end1 - start1
                n2 = end2 - start2

                sub_matrix = similarity_matrix[start1:end1, start2:end2]
                mean_val, std_val, min_val, max_val = get_stats(sub_matrix, exclude_diag=False)
                stats_table.add_row(
                    f"ID {gid1} vs ID {gid2} (Inter)",
                    f"{n1}x{n2} ({n1 * n2} pairs)",
                    f"{mean_val:.4f}",
                    f"{std_val:.4f}",
                    f"{min_val:.4f}",
                    f"{max_val:.4f}",
                )

    console.print("\n")
    console.print(Panel(stats_table, border_style="cyan", expand=False))

    # Visualization Setup
    # Dynamically scale plot size based on number of occurrences
    fig_size = max(10, total_occs * 0.35)
    # Cap size at 25 for extremely large sets, and skip tick labels if too large
    fig_size = min(fig_size, 25)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=300)

    # Plot similarity heatmap
    # Using vmin/vmax to focus on positive similarity variance if desired, or auto-scale based on values.
    # We will use the minimum value in the similarity matrix as vmin (capped at 0.0 minimum to prevent massive stretch)
    vmin = max(0.0, float(np.min(similarity_matrix)) - 0.05)
    im = ax.imshow(similarity_matrix, cmap=args.cmap, vmin=vmin, vmax=1.0)

    # Draw grid divider lines to segment the IDs
    for gid in target_ids[:-1]:
        _, end = id_indices[gid]
        ax.axvline(x=end - 0.5, color="black", linestyle="--", linewidth=2.0, alpha=0.8)
        ax.axhline(y=end - 0.5, color="black", linestyle="--", linewidth=2.0, alpha=0.8)

    # Set up ticks
    ax.set_xticks(np.arange(total_occs))
    ax.set_yticks(np.arange(total_occs))

    # Only show labels if there aren't too many occurrences
    if total_occs <= 60:
        ax.set_xticklabels(all_labels, rotation=90, ha="right", fontsize=max(4, 12 - total_occs // 5))
        ax.set_yticklabels(all_labels, fontsize=max(4, 12 - total_occs // 5))
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        console.print("[yellow]Warning: Too many occurrences (>60). Skipping individual tick labels on heatmap.[/yellow]")

    # Annotate cell values if the total occurrences are very small
    if total_occs <= 20:
        for i in range(total_occs):
            for j in range(total_occs):
                val = similarity_matrix[i, j]
                # Choose text color dynamically for contrast
                color = "white" if val < (vmin + (1.0 - vmin) / 2.0) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8, weight="bold")

    # Add background boxes/text annotations for regions on the heatmap
    for gid in target_ids:
        start, end = id_indices[gid]
        n = end - start
        if n > 2:  # Only draw text overlay if block is large enough to avoid clutter
            center = start + n / 2.0 - 0.5
            ax.text(
                center,
                center,
                f"Intra ID {gid}",
                color="black",
                fontsize=max(8, min(14, int(n * 0.8))),
                ha="center",
                va="center",
                alpha=0.35,
                weight="bold",
            )

    # For K == 2, we can also label the Inter block for backward compatibility/clarity
    if len(target_ids) == 2:
        gid1, gid2 = target_ids[0], target_ids[1]
        start1, end1 = id_indices[gid1]
        start2, end2 = id_indices[gid2]
        n1 = end1 - start1
        n2 = end2 - start2
        center1 = start1 + n1 / 2.0 - 0.5
        center2 = start2 + n2 / 2.0 - 0.5
        # Inter label top-right
        ax.text(center2, center1, "Inter Similarity", color="black", fontsize=12, ha="center", va="center", alpha=0.35, weight="bold", style="italic")
        # Inter label bottom-left
        ax.text(center1, center2, "Inter Similarity", color="black", fontsize=12, ha="center", va="center", alpha=0.35, weight="bold", style="italic")

    # Title & Colorbar
    ids_title_str = ", ".join(target_ids)
    plt.title(
        f"Pairwise Embedding Cosine Similarity Heatmap\nGlobal IDs: {ids_title_str} ({total_occs} total occurrences)",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Cosine Similarity", rotation=-90, va="bottom", fontsize=11, weight="bold")

    # Show the interactive plot if requested
    if args.show:
        console.print("\n[bold yellow]Opening interactive plot viewer...[/bold yellow]")
        plt.show()

    # Save visualization plot
    output_path = Path(args.output_plot)
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()

    console.print(f"\n[bold green]Success![/bold green] Saved similarity heatmap to: [cyan]{output_path.resolve()}[/cyan]\n")


if __name__ == "__main__":
    main()
