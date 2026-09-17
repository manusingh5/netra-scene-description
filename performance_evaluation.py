import os
import time
import threading
import psutil

from pipeline.inference import SequentialInference


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")

# How often RAM is checked during inference
RAM_SAMPLE_INTERVAL = 0.1


class MemoryMonitor:
    """
    Continuously monitors RAM used by the current Python process.
    """

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_ram_mb = 0.0
        self._stop_event = threading.Event()
        self._thread = None

    def _monitor(self):

        while not self._stop_event.is_set():

            try:
                ram_mb = (
                    self.process.memory_info().rss
                    / (1024 * 1024)
                )

                if ram_mb > self.peak_ram_mb:
                    self.peak_ram_mb = ram_mb

            except Exception:
                pass

            time.sleep(RAM_SAMPLE_INTERVAL)

    def start(self):

        self.peak_ram_mb = (
            self.process.memory_info().rss
            / (1024 * 1024)
        )

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )

        self._thread.start()

    def stop(self):

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join()

        # One final measurement
        try:

            final_ram = (
                self.process.memory_info().rss
                / (1024 * 1024)
            )

            self.peak_ram_mb = max(
                self.peak_ram_mb,
                final_ram
            )

        except Exception:
            pass

        return self.peak_ram_mb


def get_current_ram_mb():

    process = psutil.Process(
        os.getpid()
    )

    return (
        process.memory_info().rss
        / (1024 * 1024)
    )


def evaluate_video(video_name):

    video_path = os.path.join(
        UPLOADS_DIR,
        video_name
    )

    if not os.path.exists(video_path):

        print(
            f"Video not found: {video_path}"
        )

        return None


    print("\n" + "=" * 60)

    print(
        f"VIDEO: {video_name}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # MEMORY BEFORE RUN
    # --------------------------------------------------------

    ram_before = get_current_ram_mb()


    # --------------------------------------------------------
    # START MEMORY MONITOR
    # --------------------------------------------------------

    memory_monitor = MemoryMonitor()

    memory_monitor.start()


    # --------------------------------------------------------
    # START TIMER
    # --------------------------------------------------------

    start_time = time.perf_counter()


    try:

        inference = SequentialInference()

        descriptors = inference.run(
            video_path,
            fps=0.5
        )

    except Exception as error:

        print(
            f"\nERROR while processing {video_name}:"
        )

        print(error)

        memory_monitor.stop()

        return None


    # --------------------------------------------------------
    # END TIMER
    # --------------------------------------------------------

    end_time = time.perf_counter()


    # --------------------------------------------------------
    # STOP MEMORY MONITOR
    # --------------------------------------------------------

    peak_ram = memory_monitor.stop()


    # --------------------------------------------------------
    # CALCULATIONS
    # --------------------------------------------------------

    total_time = (
        end_time - start_time
    )

    total_frames = len(
        descriptors
    )


    if total_frames > 0:

        time_per_frame = (
            total_time
            / total_frames
        )

        effective_fps = (
            total_frames
            / total_time
        )

    else:

        time_per_frame = 0.0

        effective_fps = 0.0


    peak_ram_increase = max(
        0.0,
        peak_ram - ram_before
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(
        "\nPERFORMANCE RESULTS"
    )

    print(
        "-" * 40
    )


    print(
        f"Processed Frames : "
        f"{total_frames}"
    )


    print(
        f"Total Time       : "
        f"{total_time:.2f} sec"
    )


    print(
        f"Time per Frame   : "
        f"{time_per_frame:.2f} sec"
    )


    print(
        f"Effective FPS    : "
        f"{effective_fps:.3f}"
    )


    print(
        f"RAM Before       : "
        f"{ram_before:.2f} MB"
    )


    print(
        f"Peak RAM Usage   : "
        f"{peak_ram:.2f} MB"
    )


    print(
        f"Peak RAM Increase: "
        f"{peak_ram_increase:.2f} MB"
    )


    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    try:

        inference.unload_perception_models()

    except Exception:

        pass


    return {

        "video": video_name,

        "frames": total_frames,

        "total_time": total_time,

        "time_per_frame": time_per_frame,

        "fps": effective_fps,

        "ram_before": ram_before,

        "peak_ram": peak_ram,

        "peak_ram_increase": peak_ram_increase
    }


def main():

    print(
        "=" * 60
    )

    print(
        "NETRA SYSTEM PERFORMANCE EVALUATION"
    )

    print(
        "=" * 60
    )


    videos = [

        "test_video1.mp4",

        "test_video2.mp4"
    ]


    results = []


    for video in videos:

        result = evaluate_video(
            video
        )

        if result is not None:

            results.append(
                result
            )


    if not results:

        print(
            "\nNo videos were successfully evaluated."
        )

        return


    # --------------------------------------------------------
    # COMBINED PERFORMANCE
    # --------------------------------------------------------

    total_frames = sum(

        result["frames"]

        for result in results
    )


    total_time = sum(

        result["total_time"]

        for result in results
    )


    if total_frames > 0:

        overall_time_per_frame = (
            total_time
            / total_frames
        )

        overall_fps = (
            total_frames
            / total_time
        )

    else:

        overall_time_per_frame = 0.0

        overall_fps = 0.0


    # Highest RAM observed across all videos
    overall_peak_ram = max(

        result["peak_ram"]

        for result in results
    )


    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n")

    print(
        "=" * 60
    )

    print(
        "FINAL SYSTEM PERFORMANCE"
    )

    print(
        "=" * 60
    )


    print(
        f"Total Processed Frames : "
        f"{total_frames}"
    )


    print(
        f"Total Processing Time  : "
        f"{total_time:.2f} sec"
    )


    print(
        f"Average Time / Frame   : "
        f"{overall_time_per_frame:.2f} sec"
    )


    print(
        f"Effective FPS          : "
        f"{overall_fps:.3f}"
    )


    print(
        f"Peak RAM Usage         : "
        f"{overall_peak_ram:.2f} MB"
    )


    print(
        f"Peak RAM Usage         : "
        f"{overall_peak_ram / 1024:.2f} GB"
    )


    print(
        "\nNOTE:"
    )

    print(
        "Peak RAM is the maximum RSS memory observed "
        "during inference."
    )


    print(
        "\nEvaluation complete."
    )


if __name__ == "__main__":

    main()