"""Disk-backed decoding and FFmpeg encoding with the original audio timeline."""
from fractions import Fraction
from pathlib import Path
import subprocess

import av
import imageio_ffmpeg
import numpy as np


def decode_video(path, directory):
    raw_path = Path(directory) / "source.rgb"
    count, first_time, previous_time = 0, None, None
    with av.open(str(path)) as container, raw_path.open("wb") as raw:
        if not container.streams.video:
            raise ValueError("輸入檔案沒有影片軌。")
        stream = container.streams.video[0]
        fps = stream.average_rate
        if not fps or fps <= 0:
            raise ValueError("無法辨識 FPS，請先轉為固定 FPS 影片。")
        width, height = stream.codec_context.width, stream.codec_context.height
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("影片缺少時間戳，請先轉為固定 FPS 影片。")
            time = frame.pts * frame.time_base
            tolerance = frame.time_base * Fraction(11, 10)
            if first_time is None:
                first_time = time
            if (previous_time is not None and time <= previous_time) or abs(
                    time - first_time - Fraction(count, 1) / fps) > tolerance:
                raise ValueError("偵測到可變 FPS (VFR)，請先用 FFmpeg 轉為固定 FPS 再執行。")
            if frame.width != width or frame.height != height:
                raise ValueError("不支援影片中途變更畫面尺寸。")
            frame.to_ndarray(format="rgb24").tofile(raw)
            previous_time = time
            count += 1
        audio = len(container.streams.audio)
    if count < 2:
        raise ValueError("影片至少需要 2 幀。")
    frames = np.memmap(raw_path, dtype=np.uint8, mode="r", shape=(count, height, width, 3))
    return frames, {"frame_count": count, "width": width, "height": height,
                    "fps": str(fps), "first_video_timestamp": str(first_time),
                    "audio_streams": audio, "duration_seconds": float(count / fps)}


def encode_video(path, frames, source, info):
    width, height, fps = info["width"], info["height"], info["fps"]
    # copyts keeps the source audio offsets; subtract the first video PTS so the
    # reconstructed video and source audio use the same zero point.
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
               "-nostdin", "-y", "-copyts", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", fps, "-i", "pipe:0",
               "-itsoffset", str(-float(Fraction(info["first_video_timestamp"]))),
               "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
               "-c:v", "libx264", "-crf", "18", "-preset", "fast",
               "-pix_fmt", "yuv420p" if width % 2 == height % 2 == 0 else "yuv444p",
               "-c:a", "aac", "-b:a", "192k", "-t", str(info["duration_seconds"]),
               "-movflags", "+faststart", str(path)]
    # stderr goes to disk to avoid pipe deadlocks on long encoder errors.
    with Path(path).with_suffix(".ffmpeg.log").open("w+b") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for frame in frames:
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("FFmpeg 編碼失敗。")
        except BaseException as error:
            try:
                process.stdin.close()
            except OSError:
                pass
            process.wait()
            log.seek(0)
            message = log.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"影片輸出失敗：{message[-4000:]}") from error
    Path(path).with_suffix(".ffmpeg.log").unlink()
