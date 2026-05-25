from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, cast

import numpy as np

_face_recognition_import_error: Exception | None = None
_face_recognition_module = None

try:
    import face_recognition as _face_recognition_module  # pyright: ignore[reportMissingImports]
except Exception as exc:  # pragma: no cover - exercised only when dependency is absent.
    _face_recognition_import_error = exc


class _FaceRecognitionAPI(Protocol):
    def face_locations(self, image: np.ndarray, model: str = "hog") -> list[FaceLocation]: ...

    def face_encodings(
        self,
        image: np.ndarray,
        known_face_locations: list[FaceLocation] | None = None,
        num_jitters: int = 1,
    ) -> list[np.ndarray]: ...

    def face_distance(
        self,
        face_encodings: list[np.ndarray],
        face_to_compare: np.ndarray,
    ) -> np.ndarray: ...


face_recognition: _FaceRecognitionAPI | None = cast(
    _FaceRecognitionAPI | None,
    _face_recognition_module,
)
FACE_RECOGNITION_IMPORT_ERROR = _face_recognition_import_error


FaceLocation = tuple[int, int, int, int]  # top, right, bottom, left


@dataclass
class ReferenceFace:
    embedding: np.ndarray
    location: FaceLocation
    face_count: int


@dataclass
class PhotoMatch:
    result_id: str
    filename: str
    image_bytes: bytes
    mime_type: str
    matched: bool
    face_count: int
    best_distance: Optional[float]
    best_confidence: float
    low_confidence: bool
    best_face_location: Optional[FaceLocation] = None
    error: Optional[str] = None


class FaceBackendUnavailable(RuntimeError):
    """Raised when the local face-recognition backend is not installed."""


def is_backend_available() -> bool:
    return face_recognition is not None


def backend_install_message() -> str:
    if FACE_RECOGNITION_IMPORT_ERROR is None:
        return "The face recognition backend is ready."
    return (
        "The face_recognition package could not be imported. For local full matching, "
        "install requirements-local.txt and restart Streamlit. On Windows, dlib may "
        "require CMake and Microsoft Visual C++ Build Tools. On Streamlit Community "
        "Cloud this backend may be unavailable due to dlib build limits."
    )


def _require_backend() -> None:
    if face_recognition is None:
        raise FaceBackendUnavailable(backend_install_message())


def _get_backend() -> _FaceRecognitionAPI:
    _require_backend()
    return cast(_FaceRecognitionAPI, face_recognition)


def _largest_face_index(locations: list[FaceLocation]) -> int:
    areas: list[int] = []
    for top, right, bottom, left in locations:
        areas.append(max(0, right - left) * max(0, bottom - top))
    return int(np.argmax(areas))


def distance_to_confidence(distance: Optional[float], threshold: float) -> float:
    """Convert face distance into a friendly 0-1 score.

    This is a display score, not a biometric certainty. At the chosen match threshold,
    confidence is 0.50; stronger matches approach 1.00.
    """
    if distance is None:
        return 0.0

    threshold = max(float(threshold), 1e-6)
    distance = float(distance)

    if distance <= threshold:
        score = 0.5 + ((threshold - distance) / threshold) * 0.5
    else:
        upper_bound = max(1.0, threshold + 0.4)
        score = 0.5 - min(0.5, ((distance - threshold) / (upper_bound - threshold)) * 0.5)

    return float(np.clip(score, 0.0, 1.0))


def build_reference_face(
    image_array: np.ndarray,
    *,
    detection_model: str = "hog",
    num_jitters: int = 1,
) -> ReferenceFace:
    """Detect the reference face and return its embedding.

    If the image contains several faces, EchoLens uses the largest detected face because
    the reference upload is expected to be a clear photo of the target person.
    """
    backend = _get_backend()

    locations = backend.face_locations(image_array, model=detection_model)
    if not locations:
        raise ValueError("No face was detected in the reference image.")

    encodings = backend.face_encodings(
        image_array,
        known_face_locations=locations,
        num_jitters=num_jitters,
    )
    if not encodings:
        raise ValueError("A face was detected, but an embedding could not be generated.")

    largest_index = _largest_face_index(locations)
    return ReferenceFace(
        embedding=np.asarray(encodings[largest_index]),
        location=locations[largest_index],
        face_count=len(locations),
    )


def scan_event_photo(
    *,
    result_id: str,
    filename: str,
    image_bytes: bytes,
    mime_type: str,
    image_array: np.ndarray,
    reference: ReferenceFace,
    match_threshold: float,
    low_confidence_threshold: float,
    detection_model: str = "hog",
    num_jitters: int = 1,
) -> PhotoMatch:
    """Compare every detected face in one event photo with the reference embedding."""
    backend = _get_backend()

    locations = backend.face_locations(image_array, model=detection_model)
    if not locations:
        return PhotoMatch(
            result_id=result_id,
            filename=filename,
            image_bytes=image_bytes,
            mime_type=mime_type,
            matched=False,
            face_count=0,
            best_distance=None,
            best_confidence=0.0,
            low_confidence=False,
            error=None,
        )

    encodings = backend.face_encodings(
        image_array,
        known_face_locations=locations,
        num_jitters=num_jitters,
    )
    if not encodings:
        return PhotoMatch(
            result_id=result_id,
            filename=filename,
            image_bytes=image_bytes,
            mime_type=mime_type,
            matched=False,
            face_count=len(locations),
            best_distance=None,
            best_confidence=0.0,
            low_confidence=False,
            error="Faces were detected, but embeddings could not be generated.",
        )

    distances = backend.face_distance(encodings, reference.embedding)
    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])
    confidence = distance_to_confidence(best_distance, match_threshold)
    matched = best_distance <= match_threshold

    return PhotoMatch(
        result_id=result_id,
        filename=filename,
        image_bytes=image_bytes,
        mime_type=mime_type,
        matched=matched,
        face_count=len(locations),
        best_distance=best_distance,
        best_confidence=confidence,
        low_confidence=matched and confidence < low_confidence_threshold,
        best_face_location=locations[best_index],
        error=None,
    )
