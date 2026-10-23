"""Small, model-free checks: python -m unittest -v."""
from fractions import Fraction
from pathlib import Path
import json
import subprocess
import tempfile
import unittest

import av
import imageio_ffmpeg
import numpy as np

from src.core import (add_prediction, condition_window, inference_size,
                  output_frame, removal_mask, windows)
from src.cli import main, parse_args
from src.ldmvfi import midpoint_conditions
from src.video_io import decode_video, encode_video


class FrameChecks(unittest.TestCase):
    def test_patterns(self):
        self.assertEqual(np.flatnonzero(removal_mask(10, 3, "gap")).tolist(), [1, 2, 3, 5, 6, 7])
        self.assertEqual(np.flatnonzero(removal_mask(10, 3, "every")).tolist(), [2, 5, 8])
        self.assertEqual(np.flatnonzero(removal_mask(12, 2, m=5)).tolist(), [3, 4, 8, 9])
        self.assertEqual(np.flatnonzero(removal_mask(12, 1, m=4)).tolist(), [3, 7])
        np.testing.assert_array_equal(removal_mask(20, 3), removal_mask(20, 3, m=4))
        for m in (-1, 0, 1, 2):
            with self.assertRaises(ValueError):
                removal_mask(12, 2, m=m)
        with self.assertRaises(ValueError):
            removal_mask(12, 3, "every", m=4)
        for mode in ("gap", "every"):
            for count in (2, 3, 4, 10, 41):
                mask = removal_mask(count, 2, mode)
                self.assertFalse(mask[0] or mask[-1])
        for n, mode in ((0, "gap"), (1, "every"), (-2, "gap")):
            with self.assertRaises(ValueError):
                removal_mask(10, n, mode)

    def test_windows_padding_and_no_ground_truth_leak(self):
        for count in (2, 32, 33, 34, 58, 59, 100):
            coverage = np.zeros(count)
            for start, end in windows(count, 33):
                coverage[start:end] += 1
            self.assertTrue((coverage > 0).all())
            self.assertLessEqual(coverage.max(), 2)
        frames = np.arange(9 * 16 * 16 * 3, dtype=np.uint8).reshape(9, 16, 16, 3)
        mask = removal_mask(9, 3, "gap")
        altered = frames.copy()
        altered[mask] = 99
        video1, masks1 = condition_window(frames, mask, 0, 9, 17, (16, 16))
        video2, masks2 = condition_window(altered, mask, 0, 9, 17, (16, 16))
        np.testing.assert_array_equal(np.array(video1), np.array(video2))
        np.testing.assert_array_equal(np.array(masks1), np.array(masks2))
        self.assertEqual(len(video1), 17)

    def test_merge_and_unchanged_frames(self):
        sums, counts = np.zeros((2, 4, 4, 3), np.float32), np.zeros(2, np.uint32)
        add_prediction(sums, counts, 1, np.full((4, 4, 3), .2))
        add_prediction(sums, counts, 1, np.full((4, 4, 3), .8))
        np.testing.assert_allclose(sums[1] / counts[1], .5)
        original = np.zeros((8, 8, 3), np.uint8)
        self.assertIs(output_frame(original, False), original)
        self.assertEqual(output_frame(original, True, np.full((4, 4, 3), 127, np.uint8)).shape, original.shape)
        with self.assertRaises(ValueError):
            add_prediction(sums, counts, 1, np.full((4, 4, 3), np.nan))
        self.assertEqual(inference_size(1920, 1080, 512), (512, 288))

    def test_invalid_parameters(self):
        with tempfile.NamedTemporaryFile() as source:
            for arguments in (("--n", "0"), ("--n", "1", "--mode", "every"),
                              ("--n", "3", "--window-size", "32"),
                              ("--n", "3", "--model", "rife", "--model-size", "14b"),
                              ("--n", "3", "--model", "ldmvfi"),
                              ("--n", "1", "--model", "ldmvfi", "--steps", "30"),
                              ("--n", "2", "--m", "2", "--model", "rife")):
                with self.assertRaises(SystemExit):
                    parse_args(["--input", source.name, *arguments])

    def test_ldmvfi_defaults_and_no_ground_truth_leak(self):
        with tempfile.NamedTemporaryFile() as source:
            args = parse_args(["--input", source.name, "--n", "1"])
            self.assertEqual((args.model, args.steps, args.max_side, args.m), ("ldmvfi", 200, 384, 2))
            args = parse_args(["--input", source.name, "--n", "1", "--m", "5"])
            self.assertEqual(args.m, 5)
            args = parse_args(["--input", source.name, "--n", "3", "--model", "vace"])
            self.assertEqual((args.steps, args.max_side), (30, 512))
            parse_args(["--input", source.name, "--n", "3", "--mode", "every"])
        frames = np.zeros((5, 64, 32, 3), np.uint8)
        frames[0], frames[2], frames[4] = 23, 78, 211
        mask = removal_mask(5, 1, "gap")
        first = midpoint_conditions(frames, mask, 1, (32, 64))
        frames[mask] = 255
        second = midpoint_conditions(frames, mask, 1, (32, 64))
        for name, value in (("prev_frame", 23), ("next_frame", 78)):
            self.assertEqual(first[name].shape, (256, 256, 3))
            self.assertTrue((first[name] == value).all())
            np.testing.assert_array_equal(first[name], second[name])
        with self.assertRaises(ValueError):
            midpoint_conditions(frames, removal_mask(5, 3, "gap"), 1, (32, 64))


class VideoChecks(unittest.TestCase):
    def test_fractional_fps_audio_and_noop_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=30000/1001",
                "-itsoffset", "0.1", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                "-t", "1.067733333", "-c:v", "libx264", "-c:a", "aac", str(source)], check=True)
            frames, info = decode_video(source, root)
            self.assertEqual(info["fps"], "30000/1001")
            self.assertEqual(info["frame_count"], 32)
            self.assertEqual(info["audio_streams"], 1)
            target = root / "encoded.mp4"
            encode_video(target, frames, source, info)
            with av.open(str(source)) as original, av.open(str(target)) as encoded:
                self.assertEqual(encoded.streams.video[0].average_rate, Fraction(30000, 1001))
                self.assertEqual(len(list(encoded.decode(video=0))), 32)
                a0, a1 = original.streams.audio[0], encoded.streams.audio[0]
                self.assertLess(abs(float(a0.start_time * a0.time_base - a1.start_time * a1.time_base)), .05)
                self.assertLess(abs(float(a0.duration * a0.time_base - a1.duration * a1.time_base)), .05)
            del frames
            # Two frames are both endpoints: exercises the CLI without downloads.
            two = root / "two.mp4"
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", str(source),
                            "-frames:v", "2", "-an", str(two)], check=True)
            main(["--input", str(two), "--n", "1", "--device", "cpu", "--output-dir", str(root / "out")])
            result = next((root / "out").iterdir())
            self.assertEqual(json.loads((result / "run.json").read_text())["status"], "complete")
            self.assertTrue((result / "restored.mp4").is_file())

    def test_vfr_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "vfr.mp4"
            with av.open(str(path), mode="w") as output:
                stream = output.add_stream("libx264", rate=30)
                stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
                for pts in (0, 33, 66, 133, 166, 199):
                    frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="rgb24")
                    frame.pts, frame.time_base = pts, Fraction(1, 1000)
                    output.mux(stream.encode(frame))
                output.mux(stream.encode())
            with self.assertRaisesRegex(ValueError, "VFR"):
                decode_video(path, root)


if __name__ == "__main__":
    unittest.main()
