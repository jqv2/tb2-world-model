"""
Evaluation metrics:
    - Per-frame mean keypoint error (teacher forcing): Given ground truth frame t,
      how far is the predicted frame t+1 from actual frame t+1?
    - Per-problem accumulated error (autoregressive): Given only the first frame,
      predict the full sequence and calculate divergence from ground truth over time.
"""

import numpy as np


def mean_keypoint_error(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    confidence: np.ndarray | None = None,
    confidence_threshold: float = 0.3,
) -> float:
    """
    Mean Euclidean distance between predicted and ground truth keypoints,
    averaged over all valid keypoints.

    Args:
        predicted: (17, 2) predicted keypoint positions.
        ground_truth: (17, 2) ground truth keypoint positions.
        confidence: (17,) confidence scores for ground truth keypoints.
            If provided, only keypoints above confidence_threshold are included.
        confidence_threshold: Minimum confidence to include a keypoint.

    Returns:
        Mean Euclidean distance (in coordinate units) across valid keypoints,
        or np.nan if no valid keypoints exist.
    """
    if confidence is not None:
        mask = confidence >= confidence_threshold
    else:
        mask = np.ones(len(predicted), dtype=bool)

    if not mask.any():
        return np.nan

    distances = np.linalg.norm(predicted[mask] - ground_truth[mask], axis=1)
    return float(distances.mean())


def per_frame_errors(
    predicted_frames: list[np.ndarray],
    gt_frames: list[np.ndarray],
    gt_confidences: list[np.ndarray] | None = None,
    confidence_threshold: float = 0.3,
) -> np.ndarray:
    """
    Per-frame mean keypoint error between predicted and ground truth poses.

    predicted_frames[i] is compared against gt_frames[i+1]. The caller is
    responsible for generating predictions under the desired conditions:
        - Teacher forcing: each prediction uses ground truth as input.
        - Autoregressive: each prediction uses the model's own prior output.

    Args:
        predicted_frames: List of N-1 predicted poses, each (17, 2).
        gt_frames: List of N ground truth poses, each (17, 2).
        gt_confidences: List of N confidence arrays, each (17,). Optional.
        confidence_threshold: Minimum confidence to include a keypoint.

    Returns:
        Array of N-1 per-frame errors.
    """
    n_predictions = len(predicted_frames)
    assert n_predictions == len(gt_frames) - 1, (
        f"Expected {len(gt_frames) - 1} predictions for {len(gt_frames)} GT frames, "
        f"got {n_predictions}"
    )

    errors = np.empty(n_predictions)
    for i in range(n_predictions):
        conf = gt_confidences[i + 1] if gt_confidences is not None else None
        errors[i] = mean_keypoint_error(
            predicted_frames[i], gt_frames[i + 1], conf, confidence_threshold
        )
    return errors


def summarize_teacher_forcing(errors: np.ndarray) -> dict:
    """
    Compute summary statistics for a sequence of per-frame errors.

    Args:
        errors: Array of per-frame errors (may contain NaN for skipped frames).

    Returns:
        Dict with mean, median, std, max, and count of valid frames.
    """
    valid = errors[~np.isnan(errors)]
    if len(valid) == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan, "max": np.nan, "n_valid": 0}
    return {
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "std": float(valid.std()),
        "max": float(valid.max()),
        "n_valid": len(valid),
    }
    
def summarize_autoregressive(errors: np.ndarray) -> dict:
    """
    Summarize autoregressive errors by reporting error at progression milestones.

    Shows how error evolves as the model predicts further into the sequence,
    which reveals whether predictions diverge over time.

    Args:
        errors: Array of per-frame errors in sequence order (may contain NaN).

    Returns:
        Dict with error at 25%, 50%, 75%, and 100% through the sequence.
    """
    valid = errors[~np.isnan(errors)]
    if len(valid) == 0:
        return {"p25": np.nan, "p50": np.nan, "p75": np.nan, "p100": np.nan}
    indices = [len(valid) // 4, len(valid) // 2, 3 * len(valid) // 4, len(valid) - 1]
    return {
        "p25": float(valid[indices[0]]),
        "p50": float(valid[indices[1]]),
        "p75": float(valid[indices[2]]),
        "p100": float(valid[indices[3]]),
    }