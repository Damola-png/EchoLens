from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import pandas as pd

from .face_matcher import PhotoMatch
from .image_utils import safe_filename


@dataclass
class ReliabilityStats:
    total_images: int
    total_faces: int
    matched_photos: int
    unmatched_photos: int
    average_confidence: float
    low_confidence_matches: int


def summarize_results(results: list[PhotoMatch]) -> ReliabilityStats:
    matched = [result for result in results if result.matched]
    average_confidence = (
        sum(result.best_confidence for result in matched) / len(matched) if matched else 0.0
    )

    return ReliabilityStats(
        total_images=len(results),
        total_faces=sum(result.face_count for result in results),
        matched_photos=len(matched),
        unmatched_photos=len(results) - len(matched),
        average_confidence=average_confidence,
        low_confidence_matches=sum(result.low_confidence for result in matched),
    )


def results_to_dataframe(
    results: list[PhotoMatch],
    review_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    review_labels = review_labels or {}
    rows = []

    for result in results:
        rows.append(
            {
                "result_id": result.result_id,
                "filename": result.filename,
                "matched": result.matched,
                "faces_detected": result.face_count,
                "best_distance": result.best_distance,
                "confidence": round(result.best_confidence, 4),
                "low_confidence": result.low_confidence,
                "manual_review": review_labels.get(result.result_id, "Unreviewed"),
                "error": result.error,
            }
        )

    return pd.DataFrame(rows)


def build_matches_zip(results: list[PhotoMatch]) -> bytes:
    buffer = BytesIO()
    seen_names: dict[str, int] = {}

    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        for result in results:
            if not result.matched:
                continue

            clean_name = safe_filename(result.filename)
            count = seen_names.get(clean_name, 0)
            seen_names[clean_name] = count + 1

            if count:
                stem, dot, suffix = clean_name.rpartition(".")
                clean_name = f"{stem}_{count}.{suffix}" if dot else f"{clean_name}_{count}"

            archive.writestr(f"matched_photos/{clean_name}", result.image_bytes)

    buffer.seek(0)
    return buffer.getvalue()


def confidence_histogram(results: list[PhotoMatch]):
    matched_scores = [result.best_confidence * 100 for result in results if result.matched]
    fig, ax = plt.subplots(figsize=(7, 3.4))

    if matched_scores:
        ax.hist(matched_scores, bins=8, color="#5B4CFF", edgecolor="#FFFFFF", linewidth=1.2)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence score")
        ax.set_ylabel("Matched photos")
    else:
        ax.text(
            0.5,
            0.5,
            "No matched photos yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#667085",
        )
        ax.set_axis_off()

    ax.set_title("Match confidence distribution", loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6ECF5", linewidth=0.8)
    fig.tight_layout()
    return fig
