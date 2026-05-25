from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.face_matcher import (
    FaceBackendUnavailable,
    PhotoMatch,
    ReferenceFace,
    backend_install_message,
    build_reference_face,
    is_backend_available,
    scan_event_photo,
)
from src.image_utils import SUPPORTED_IMAGE_TYPES, load_image_from_bytes
from src.reporting import (
    build_matches_zip,
    confidence_histogram,
    results_to_dataframe,
    summarize_results,
)


BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "EchoLens Logo.png"


def get_page_icon():
    if LOGO_PATH.exists():
        return Image.open(LOGO_PATH)
    return None


st.set_page_config(
    page_title="EchoLens",
    page_icon=get_page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Manrope:wght@400;500;700&display=swap');

        :root {
            --echolens-navy: #102a43;
            --echolens-muted: #52606d;
            --echolens-border: #d3dfeb;
            --echolens-blue: #0ea5e9;
            --echolens-coral: #ff7a59;
            --echolens-panel: #ffffff;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 8%, rgba(14, 165, 233, 0.14), transparent 35%),
                radial-gradient(circle at 92% 18%, rgba(255, 122, 89, 0.13), transparent 34%),
                linear-gradient(180deg, #f7fbff 0%, #eef5ff 42%, #f9fcff 100%);
            color: var(--echolens-navy);
            font-family: 'Manrope', 'Segoe UI', sans-serif;
        }

        .block-container {
            max-width: 1220px;
            padding-top: 1.25rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7fbff 0%, #eef5ff 100%);
            border-right: 1px solid var(--echolens-border);
        }

        .echolens-hero {
            border: 1px solid var(--echolens-border);
            background: linear-gradient(120deg, rgba(255, 255, 255, 0.88), rgba(250, 254, 255, 0.96));
            backdrop-filter: blur(5px);
            border-radius: 18px;
            padding: 1.35rem 1.45rem;
            box-shadow: 0 16px 42px rgba(16, 42, 67, 0.11);
            margin-bottom: 1rem;
            animation: echolensFadeIn 460ms ease-out;
        }

        .echolens-title {
            font-size: clamp(2.2rem, 5vw, 4.25rem);
            line-height: 0.98;
            font-weight: 800;
            font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
            margin: 0;
            letter-spacing: -0.02em;
            color: var(--echolens-navy);
        }

        .echolens-subtitle {
            max-width: 760px;
            color: var(--echolens-muted);
            font-size: 1.02rem;
            line-height: 1.65;
            margin-top: 0.55rem;
        }

        .echolens-section-title {
            font-size: 1.15rem;
            font-weight: 760;
            font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
            color: var(--echolens-navy);
            margin-bottom: 0.35rem;
        }

        .echolens-muted {
            color: var(--echolens-muted);
            font-size: 0.94rem;
            line-height: 1.5;
        }

        .status-ready {
            border-left: 4px solid #17b26a;
            background: #ecfdf3;
            color: #05603a;
            padding: 0.8rem 0.9rem;
            border-radius: 12px;
        }

        .status-warn {
            border-left: 4px solid #f79009;
            background: #fffaeb;
            color: #7a2e0e;
            padding: 0.8rem 0.9rem;
            border-radius: 12px;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--echolens-border);
            border-radius: 14px;
            padding: 0.95rem 1rem;
            box-shadow: 0 8px 22px rgba(16, 42, 67, 0.08);
            transition: transform 180ms ease, box-shadow 180ms ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 30px rgba(16, 42, 67, 0.12);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--echolens-muted);
            font-size: 0.82rem;
        }

        div[data-testid="stMetricValue"] {
            color: var(--echolens-navy);
            font-weight: 780;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 12px;
            font-weight: 700;
            border: 1px solid rgba(14, 165, 233, 0.2);
            box-shadow: 0 10px 18px rgba(16, 42, 67, 0.1);
        }

        [data-testid="stFileUploader"] {
            border: 1px dashed #b7c7dd;
            border-radius: 12px;
            padding: 0.25rem;
            background: rgba(255, 255, 255, 0.75);
        }

        @keyframes echolensFadeIn {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 720px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .echolens-hero {
                padding: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def clear_all_data() -> None:
    keys_to_clear = [
        "scan_results",
        "review_labels",
        "reference_upload",
        "event_uploads",
        "scan_completed",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def show_sidebar() -> tuple[float, float, str, int]:
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)

    st.sidebar.markdown("### Privacy")
    st.sidebar.info(
        "EchoLens processes uploads in this browser session. It does not save faces, "
        "embeddings, or event photos to disk unless you download an export yourself."
    )

    st.sidebar.markdown("### Matching controls")
    match_threshold = st.sidebar.slider(
        "Match distance threshold",
        min_value=0.35,
        max_value=0.75,
        value=0.60,
        step=0.01,
        help="Lower values are stricter. 0.60 is a common starting point for face_recognition.",
    )
    low_confidence_threshold = st.sidebar.slider(
        "Low-confidence warning below",
        min_value=0.50,
        max_value=0.90,
        value=0.68,
        step=0.01,
        help="Matches below this displayed confidence are flagged for manual review.",
    )
    detection_model = st.sidebar.selectbox(
        "Face detector",
        options=("hog", "cnn"),
        index=0,
        help="HOG is faster on most laptops. CNN can be more accurate but is much slower without a GPU.",
    )
    num_jitters = st.sidebar.slider(
        "Embedding quality passes",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        help="Higher values can improve embeddings but make scans slower.",
    )

    st.sidebar.markdown("### Data controls")
    st.sidebar.button(
        "Clear uploaded images and face data",
        type="secondary",
        use_container_width=True,
        on_click=clear_all_data,
    )

    return match_threshold, low_confidence_threshold, detection_model, num_jitters


def show_header() -> None:
    st.markdown('<div class="echolens-hero">', unsafe_allow_html=True)
    logo_col, text_col = st.columns([0.24, 0.76], vertical_alignment="center")
    with logo_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
    with text_col:
        st.markdown('<h1 class="echolens-title">EchoLens</h1>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="echolens-subtitle">
            Upload one reference face, scan a batch of event photos, and return a focused
            gallery of the images where that person appears.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def show_backend_status() -> None:
    if is_backend_available():
        st.markdown(
            '<div class="status-ready">Face matching backend detected. EchoLens is ready to scan photos.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="status-warn">{backend_install_message()}</div>',
            unsafe_allow_html=True,
        )


def show_reference_panel(
    reference_upload: UploadedFile | None,
    detection_model: str,
    num_jitters: int,
) -> ReferenceFace | None:
    reference: ReferenceFace | None = None
    reference_image = None

    if reference_upload is None:
        st.markdown(
            '<p class="echolens-muted">Choose a clear, front-facing photo with one main person.</p>',
            unsafe_allow_html=True,
        )
        return None

    image_bytes = reference_upload.getvalue()
    try:
        reference_image, reference_array = load_image_from_bytes(image_bytes)
        st.image(reference_image, caption="Reference image", use_container_width=True)

        if is_backend_available():
            reference = build_reference_face(
                reference_array,
                detection_model=detection_model,
                num_jitters=num_jitters,
            )
            if reference.face_count > 1:
                st.warning(
                    f"{reference.face_count} faces were detected. EchoLens used the largest face as the reference."
                )
            else:
                st.success("Reference face detected and encoded.")
        else:
            st.warning("Install the face matching backend before scanning photos.")
    except Exception as exc:
        st.error(f"Reference image could not be processed: {exc}")

    return reference


def process_event_photos(
    *,
    event_uploads: Sequence[UploadedFile],
    reference: ReferenceFace,
    match_threshold: float,
    low_confidence_threshold: float,
    detection_model: str,
    num_jitters: int,
) -> list[PhotoMatch]:
    results: list[PhotoMatch] = []
    progress_bar = st.progress(0)
    status = st.empty()

    for index, uploaded_file in enumerate(event_uploads):
        status.write(f"Scanning {uploaded_file.name}...")
        result_id = f"{index + 1:04d}_{uploaded_file.name}"
        image_bytes: bytes = uploaded_file.getvalue()
        mime_type: str = uploaded_file.type or "image/jpeg"

        try:
            _, image_array = load_image_from_bytes(image_bytes)
            result = scan_event_photo(
                result_id=result_id,
                filename=uploaded_file.name,
                image_bytes=image_bytes,
                mime_type=mime_type,
                image_array=image_array,
                reference=reference,
                match_threshold=match_threshold,
                low_confidence_threshold=low_confidence_threshold,
                detection_model=detection_model,
                num_jitters=num_jitters,
            )
        except FaceBackendUnavailable as exc:
            raise exc
        except Exception as exc:
            result = PhotoMatch(
                result_id=result_id,
                filename=uploaded_file.name,
                image_bytes=image_bytes,
                mime_type=mime_type,
                matched=False,
                face_count=0,
                best_distance=None,
                best_confidence=0.0,
                low_confidence=False,
                error=str(exc),
            )

        results.append(result)
        progress_bar.progress((index + 1) / len(event_uploads))

    status.write("Scan complete.")
    return results


def show_reliability_dashboard(results: list[PhotoMatch]) -> None:
    stats = summarize_results(results)
    st.subheader("Reliability dashboard")
    metric_cols = st.columns(6)
    metric_cols[0].metric("Images uploaded", stats.total_images)
    metric_cols[1].metric("Faces detected", stats.total_faces)
    metric_cols[2].metric("Matched photos", stats.matched_photos)
    metric_cols[3].metric("Unmatched photos", stats.unmatched_photos)
    metric_cols[4].metric("Avg confidence", f"{stats.average_confidence:.0%}")
    metric_cols[5].metric("Low confidence", stats.low_confidence_matches)

    chart_col, table_col = st.columns([0.55, 0.45])
    with chart_col:
        fig = confidence_histogram(results)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with table_col:
        summary_df = pd.DataFrame(
            {
                "Status": ["Matched", "Unmatched"],
                "Photos": [stats.matched_photos, stats.unmatched_photos],
            }
        ).set_index("Status")
        st.bar_chart(summary_df)


def show_results_gallery(results: list[PhotoMatch]) -> None:
    st.subheader("Results gallery")
    matches = [result for result in results if result.matched]
    low_confidence_matches = [result for result in matches if result.low_confidence]

    if low_confidence_matches:
        st.warning(
            f"{len(low_confidence_matches)} match(es) have lower confidence. Manual review is recommended."
        )

    if not matches:
        st.info("No matching photos were found with the current threshold.")
        return

    min_confidence = st.slider(
        "Filter matched photos by minimum confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        format="%.2f",
    )
    filtered_matches = [
        result for result in matches if result.best_confidence >= min_confidence
    ]

    st.session_state.setdefault("review_labels", {})
    gallery_columns = st.columns(3)

    for index, result in enumerate(filtered_matches):
        with gallery_columns[index % 3]:
            st.image(result.image_bytes, use_container_width=True)
            st.markdown(f"**{result.filename}**")
            st.caption(
                f"Confidence: {result.best_confidence:.0%} | Distance: {result.best_distance:.3f}"
            )
            label = st.radio(
                "Manual review",
                options=("Unreviewed", "Correct", "Incorrect"),
                horizontal=True,
                key=f"review_{result.result_id}",
            )
            st.session_state["review_labels"][result.result_id] = label

    zip_bytes = build_matches_zip(filtered_matches)
    st.download_button(
        "Download visible matches as ZIP",
        data=zip_bytes,
        file_name="echolens_matched_photos.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )


def show_exports(results: list[PhotoMatch]) -> None:
    review_labels = st.session_state.get("review_labels", {})
    dataframe = results_to_dataframe(results, review_labels)
    st.download_button(
        "Download evaluation CSV",
        data=dataframe.to_csv(index=False).encode("utf-8"),
        file_name="echolens_evaluation.csv",
        mime="text/csv",
        use_container_width=True,
    )
    with st.expander("View scan details"):
        st.dataframe(dataframe, use_container_width=True, hide_index=True)


def show_explanation() -> None:
    with st.expander("How EchoLens face matching works"):
        st.markdown(
            """
            1. EchoLens detects faces in the uploaded reference photo.
            2. The largest reference face is converted into a numeric embedding.
            3. Each event photo is scanned for faces and each detected face gets its own embedding.
            4. The app compares event embeddings to the reference embedding using face distance.
            5. Photos with a distance at or below the selected threshold are marked as matches.

            Confidence is a readable score derived from distance. It is useful for ranking and
            review, but it should not be treated as a guaranteed identity claim.
            """
        )


def main() -> None:
    inject_styles()
    match_threshold, low_confidence_threshold, detection_model, num_jitters = show_sidebar()
    show_header()
    show_backend_status()

    upload_col, event_col = st.columns(2)
    with upload_col:
        st.markdown('<div class="echolens-section-title">1. Reference face</div>', unsafe_allow_html=True)
        reference_upload = st.file_uploader(
            "Upload one clear photo of the person to find",
            type=SUPPORTED_IMAGE_TYPES,
            key="reference_upload",
        )
        reference = show_reference_panel(reference_upload, detection_model, num_jitters)

    with event_col:
        st.markdown('<div class="echolens-section-title">2. Event photos</div>', unsafe_allow_html=True)
        event_uploads = st.file_uploader(
            "Upload event photos",
            type=SUPPORTED_IMAGE_TYPES,
            accept_multiple_files=True,
            key="event_uploads",
        )
        total_event_photos = len(event_uploads) if event_uploads else 0
        st.markdown(
            f'<p class="echolens-muted">{total_event_photos} photo(s) selected for scanning.</p>',
            unsafe_allow_html=True,
        )

    scan_disabled = (
        reference is None
        or not event_uploads
        or not is_backend_available()
    )
    scan_col, helper_col = st.columns([0.35, 0.65], vertical_alignment="center")
    with scan_col:
        scan_clicked = st.button(
            "Scan event photos",
            type="primary",
            use_container_width=True,
            disabled=scan_disabled,
        )
    with helper_col:
        st.markdown(
            '<p class="echolens-muted">Photos are scanned in memory and results are kept only for this session.</p>',
            unsafe_allow_html=True,
        )

    if scan_clicked:
        try:
            if reference is None:
                st.error("Upload a valid reference face before scanning event photos.")
                return
            st.session_state["scan_results"] = process_event_photos(
                event_uploads=event_uploads,
                reference=reference,
                match_threshold=match_threshold,
                low_confidence_threshold=low_confidence_threshold,
                detection_model=detection_model,
                num_jitters=num_jitters,
            )
            st.session_state["scan_completed"] = True
        except FaceBackendUnavailable as exc:
            st.error(str(exc))

    results = st.session_state.get("scan_results", [])
    if results:
        st.divider()
        show_reliability_dashboard(results)
        st.divider()
        show_results_gallery(results)
        st.divider()
        show_exports(results)

    st.divider()
    show_explanation()


if __name__ == "__main__":
    main()
