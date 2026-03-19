import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration": duration
    }


def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    motion_values = []
    hist_values = []
    times = []

    prev_gray = None
    prev_hist = None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Feature 1 : Motion magnitude using frame differencing
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion = np.mean(diff)
            motion_values.append(motion)


        # Feature 2: Histogram change
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            hist_change = np.sum(np.abs(hist - prev_hist))
            hist_values.append(hist_change)

        if prev_gray is not None and prev_hist is not None:
            times.append(frame_idx / fps)

        prev_gray = gray
        prev_hist = hist
        frame_idx += 1

    cap.release()

    return np.array(times), np.array(motion_values), np.array(hist_values)


def save_sample_frames(video_path, output_path, num_frames=3):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((idx, frame_rgb))

    cap.release()

    plt.figure(figsize=(12, 4))
    for i, (idx, frame) in enumerate(frames):
        plt.subplot(1, len(frames), i + 1)
        plt.imshow(frame)
        plt.title(f"Frame {idx}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_features(times, motion_values, hist_values, output_path):
    plt.figure(figsize=(12, 6))

    plt.plot(times, motion_values, label="Motion Magnitude")
    plt.plot(times, hist_values, label="Histogram Change")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Feature Value")
    plt.title("Behavioral Features Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_activity_summary(times, motion_values):
    if len(motion_values) == 0:
        print("No motion values extracted.")
        return

    mean_motion = np.mean(motion_values)
    high_activity = times[motion_values > mean_motion]
    low_activity = times[motion_values <= mean_motion]

    print(f"Average motion magnitude: {mean_motion:.2f}")

    if len(high_activity) > 0:
        print(f"Plausible high-activity periods around: {high_activity[:10]}")

    if len(low_activity) > 0:
        print(f"Plausible low-activity periods around: {low_activity[:10]}")


def main():
    video_path = "data/octopus_camouflage.mp4"
    os.makedirs("outputs", exist_ok=True)

    info = get_video_info(video_path)
    print("Video Information")
    print(f"FPS: {info['fps']:.2f}")
    print(f"Resolution: {info['width']} x {info['height']}")
    print(f"Total Frames: {info['total_frames']}")
    print(f"Duration: {info['duration']:.2f} seconds")

    times, motion_values, hist_values = extract_features(video_path)

    plot_features(times, motion_values, hist_values, "outputs/feature_plot.png")
    save_sample_frames(video_path, "outputs/sample_frames.png")

    print_activity_summary(times, motion_values)
    print("Plots saved in outputs/")


if __name__ == "__main__":
    main()