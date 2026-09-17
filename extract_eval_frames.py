import os
import cv2

from ground_truth import GROUND_TRUTH


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

UPLOADS_DIR = os.path.join(
    PROJECT_ROOT,
    "uploads"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "evaluation_frames"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def extract_frame(video_path, timestamp):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        timestamp * 1000
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        return None

    return frame


def main():

    print("=" * 60)
    print("NETRA - Extract Evaluation Frames")
    print("=" * 60)

    total_saved = 0

    for video_name, video_data in GROUND_TRUTH.items():

        video_path = os.path.join(
            UPLOADS_DIR,
            video_name
        )

        if not os.path.exists(video_path):

            print(
                f"\nVideo not found: {video_path}"
            )

            continue

        video_folder_name = os.path.splitext(
            video_name
        )[0]

        video_output_dir = os.path.join(
            OUTPUT_DIR,
            video_folder_name
        )

        os.makedirs(
            video_output_dir,
            exist_ok=True
        )

        for timestamp in video_data["frames"].keys():

            frame = extract_frame(
                video_path,
                timestamp
            )

            if frame is None:

                print(
                    f"Could not extract "
                    f"{video_name} at {timestamp}s"
                )

                continue

            filename = (
                f"frame_{timestamp:03d}s.jpg"
            )

            output_path = os.path.join(
                video_output_dir,
                filename
            )

            cv2.imwrite(
                output_path,
                frame
            )

            total_saved += 1

            print(
                f"Saved: "
                f"{video_folder_name}/{filename}"
            )

    print("\n" + "=" * 60)
    print(
        f"Total frames saved: {total_saved}"
    )
    print(
        f"Folder: {OUTPUT_DIR}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()