"""
Run pose extraction to get ViTPose keypoints from a video file and save to JSON.

Usage:
    Save output to inferred output filename: python scripts/run_pose_extraction.py data/raw/climb_001.MOV
    Save output to specific output filename: python scripts/run_pose_extraction.py data/raw/climb_001.MOV --output data/poses/climb_001.json
    Save output to inferred output filename + show live preview: python scripts/run_pose_extraction.py data/raw/climb_001.MOV --show
    Save output to inferred output filename + save video with pose/bbox overlay: python scripts/run_pose_extraction.py data/raw/climb_001.MOV --save-video

Output JSON structure (keypoint_names included for convenience):
    {
        "video": "climb_001.MOV",
        "fps": 30.0,
        "total_frames": 600,
        "keypoint_names": ["nose", "left_eye", ...],
        "frames": [
            {
                "frame_idx": 0,
                "box": [x1, y1, x2, y2] | null,
                "keypoints": [[x, y], ...] | null,
                "scores": [0.99, ...] | null
            },
            ...
        ]
    }
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.pose_extraction import load_models, extract_frame_pose


def extract_video_poses(video_path: Path, models: dict, show: bool = False, save_video: bool = False) -> dict:
    """
    Run pose extraction on every frame of a video.

    Args:
        video_path: Path to the input video file.
        models: Dict returned by load_models().
        show: If True, display annotated frames in a window.

    Returns:
        Dict containing metadata and per-frame pose data.
    """
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save_video:
        out_path = video_path.parent.parent / "poses" / f"{video_path.stem}_overlay.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        
    frames_data = []

    print(f"Processing {video_path.name}: {total_frames} frames @ {fps:.1f} fps")

    frame_idx = 0
    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        result = extract_frame_pose(pil_image, models)

        if result is not None:
            frame_data = {
                "frame_idx": frame_idx,
                "box": result["box"].tolist(),
                "keypoints": result["keypoints"].tolist(),
                "scores": result["scores"].tolist(),
            }
        else:
            frame_data = {
                "frame_idx": frame_idx,
                "box": None,
                "keypoints": None,
                "scores": None,
            }

        frames_data.append(frame_data)

        if writer is not None:
            display_frame = bgr_frame.copy()
            if result is not None:
                _draw_pose(display_frame, result)
            writer.write(display_frame)
            
        if show and result is not None:
            _draw_pose(bgr_frame, result)
            cv2.imshow("ViTPose", bgr_frame)
            cv2.waitKey(1)

        frame_idx += 1
        if frame_idx % 50 == 0:
            detected = sum(1 for f in frames_data if f["box"] is not None)
            print(f"  Frame {frame_idx}/{total_frames}. Person detected in {detected}/{frame_idx} frames so far.")

    cap.release()
    if show:
        cv2.destroyAllWindows()
        
    detected = sum(1 for f in frames_data if f["box"] is not None)
    print(f"Done. Person detected in {detected}/{frame_idx} frames.")
        
    if writer is not None:
        writer.release()
        print(f"Saved overlay video to {out_path}")

    return {
        "video": video_path.name,
        "fps": fps,
        "total_frames": frame_idx,
        "keypoint_names": config.COCO_KEYPOINT_NAMES,
        "frames": frames_data,
    }
    

def _draw_pose(bgr_frame: np.ndarray, result: dict) -> None:
    """
    Draw keypoints and skeleton on a BGR frame (in-place, for preview only).
    """
    keypoints = result["keypoints"]
    scores = result["scores"]
    
    # draw bounding box
    box = result["box"]
    cv2.rectangle(
        bgr_frame,
        (int(box[0]), int(box[1])),
        (int(box[2]), int(box[3])),
        (255, 0, 0), 2
    )

    for (i, j) in config.COCO_SKELETON:
        if (scores[i] > config.KEYPOINT_CONFIDENCE_THRESHOLD
                and scores[j] > config.KEYPOINT_CONFIDENCE_THRESHOLD):
            pt1 = tuple(int(v) for v in keypoints[i])
            pt2 = tuple(int(v) for v in keypoints[j])
            cv2.line(bgr_frame, pt1, pt2, (0, 255, 0), 2)

    for _, (kp, score) in enumerate(zip(keypoints, scores)):
        if score > config.KEYPOINT_CONFIDENCE_THRESHOLD:
            cv2.circle(bgr_frame, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)
            
            
def main():
    parser = argparse.ArgumentParser(description="Extract poses from a climbing video")
    parser.add_argument("video", type=Path, help="Path to input video file")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path (default: data/poses/<video_name>.json)"
    )
    parser.add_argument("--show", action="store_true", help="Show live preview")
    parser.add_argument("--device", type=str, default=None, help="cuda / mps / cpu")
    parser.add_argument("--save-video", action="store_true", help="Save annotated overlay video")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    output_path = args.output or config.POSES_DIR / f"{args.video.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    models = load_models(device=args.device)
    print(f"Using device: {models['device']}")

    data = extract_video_poses(args.video, models, show=args.show, save_video=args.save_video)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()