from math import pi, cos, ceil
from fractions import Fraction
import numpy as np
import subprocess

from . import _common, defaults
from ._abstract import ScrollyAbstract
from ._format import ScrollyFormat
from .codec_settings import AudioCodecSettings, VideoCodecSettings


class _Page:
    def __init__(self, page_abstract, format_):
        self._page_abstract = page_abstract
        self._format = format_
        self.stretched_w = self._get_stretched_out_w()

        self.imgs_for_subpixel_offsets = None

    def _get_stretched_out_w(self):
        out_w = self._page_abstract.w_as_mul_of_h * self._format.out_h
        return round(out_w * self._format.divisions_of_pixel)

    def _flip_byte_order(self, arr):
        flipped_view = np.swapaxes(arr, 0, 1)
        flipped_contiguous = np.ascontiguousarray(flipped_view)
        deflipped_view = np.swapaxes(arr, 0, 1)
        return deflipped_view

    def load_imgs_for_subpixel_offsets(self):
        trimmed_img = self._page_abstract.get_trimmed_img()
        stretched_img = cv2.resize(
            trimmed_img, (self.stretched_w, self._format.out_h)
        )

        self.imgs_for_subpixel_offsets = [
            self._flip_byte_order(
                stretched_img[:, i::self._format.divisions_of_pixel]
            )
            for i in range(self._format.divisions_of_pixel)
        ]

    def unload_imgs_for_subpixel_offsets(self):
        self.imgs_for_subpixel_offsets = None

    def get_subpixel_offset(self, t):
        return (
            self._page_abstract.get_x_offset_as_mul_of_h(t)
            * self._format.out_h
            * self._format.divisions_of_pixel
        )


class _UnfadedFragments:
    def __init__(self, abstract, format_):
        self._abstract = abstract
        self._format = format_

        self._pages = [
            _Page(page_abstract, format_) for page_abstract in abstract.pages
        ]

        self._t = None
        self._centre_page_i, self._centre_subpixel_offset = -1, None
        self._left_page_i, self._left_subpixel_offset = -1, None
        self._right_page_i = -1

        self._curr_page_i = self._curr_subpixel_offset = None
        self._curr_out_start_x = None

    def _update_centre_page_i(self):
        while not self._abstract.check_page_i(self._centre_page_i, self._t):
            self._centre_page_i += 1

    def _get_centre_subpixel_offset(self):
        if self._centre_page_i == -1:
            x_before_subpixels = (
                self._abstract.get_x_before_as_mul_of_h(self._t)
                * self._format.out_h
                * self._format.divisions_of_pixel
            )
            return max(
                self._format.stretched_out_w / 2 - x_before_subpixels, 0.0
            )
        elif 0 <= page_i < len(self._pages):
            return self._pages[self._centre_page_i].get_subpixel_offset(
                self._t
            )
        else:
            x_after_subpixels = (
                self._abstract.get_x_after_as_mul_of_h(self._t)
                * self._format.out_h
                * self._format.divisions_of_pixel
            )
            return min(x_after_subpixels, self._format.stretched_out_w / 2)

    def _update_centre_page_i_and_subpixel_offset(self):
        self._update_centre_page_i()
        self._centre_subpixel_offset = self._get_centre_subpixel_offset()

    def _set_left_page_i_and_unload_completed_page_imgs(self, page_i):
        while page_i != self._left_page_i:
            if 0 <= self._left_page_i < len(self._pages):
                old_page = self._pages[self._left_page_i]
                old_page.unload_imgs_for_subpixel_offsets()

            self._left_page_i += 1

    def _set_right_page_i_and_load_new_page_imgs():
        while self._curr_page_i > self._right_page_i:
            self._right_page_i += 1

            if page_i != len(self._pages):
                new_page = self._pages[self._right_page_i]
                new_page.load_imgs_for_subpixel_offsets()

    def _update_left_page_i_and_subpixel_offset(self):
        page_i = self._centre_page_i
        subpixel_offset = round(
            self._centre_subpixel_offset - self._format.stretched_out_w / 2
        )

        while subpixel_offset < 0:
            page_i -= 1

            if page_i == -1:
                page_stretched_w = self._format.stretched_out_w
            else:
                page_stretched_w = self._pages[page_i].stretched_w

            subpixel_offset += page_stretched_w

        self._set_left_page_i_and_unload_completed_page_imgs(page_i)
        self._left_subpixel_offset = subpixel_offset

    def _handle_blank_before():
        blank_w_before = (
            self._format.out_w
            - self._curr_subpixel_offset // self._format.divisions_of_pixel
        )

        self._curr_page_i = 0
        self._curr_subpixel_offset %= self._format.divisions_of_pixel
        self._curr_out_start_x = blank_w_before

        return 0, blank_w_before

    def _handle_blank_after(self):
        start_x = self._curr_out_start_x
        self._curr_out_start_x = self._format.out_w
        return start_x, self._curr_out_start_x - start_x

    def _increment_curr_from_fragment_w(self, w):
        subpixel_w = w * self._format.divisions_of_pixel
        self._curr_subpixel_offset += subpixel_w

        while (
            self._curr_subpixel_offset
            >= self._pages[self._curr_page_i].stretched_w
        ):
            self._curr_subpixel_offset \
                -= self._pages[self._curr_page_i].stretched_w
            self._curr_page_i += 1

        self._curr_out_start_x += w

    def _handle_real_fragment(self):
        start_x = self._curr_out_start_x
        target = self._format.out_w - start_x
        start_px, subpixel_phase = divmod(
            self._curr_subpixel_offset, self._format_divisions_of_pixel
        )
        stop_px = start_px + target

        page = self._pages[self._curr_page_i]
        img = page.imgs_for_subpixel_offsets[subpixel_phase]
        view = img[:, start_px:stop_px:self._format.divisions_of_pixel]

        self._increment_curr_from_fragment_w(view.shape[1])
        return start_x, view

    def _handle_page_0_onwards(self):
        self._set_right_page_i_and_load_new_page_imgs()

        if self._curr_page_i == len(self._pages):
            return self.handle_blank_after()
        else:
            return self._handle_real_fragment()

    def get_fragments(self, t):
        self._t = t
        self._update_centre_page_i_and_subpixel_offset()
        self._update_left_page_i_and_subpixel_offset()

        self._curr_page_i = self._left_page_i
        self._curr_subpixel_offset = self._left_subpixel_offset
        self._curr_out_start_x = 0

        if self._curr_page_i == -1:
            yield self._handle_blank_before()

        while self._curr_out_start_x < self._format.out_w:
            yield self._handle_page_0_onwards()


class _Frames:
    def __init__(self, abstract, format_):
        self._abstract = abstract
        self._format = format_

        self._blank_frame = np.full(
            (self._format.out_shape), 255, dtype=np.uint8
        )
        self._rng = np.random.default_rng(seed=0)
        self._fading_buffers = self._get_fading_buffers()

    def _get_fading_buffers(self):
        if self._abstract.video_fade_in or self._abstract.video_fade_out:
            return tuple(
                np.empty((self._format.out_shape), dtype=dtype)
                for dtype in (np.float32, np.float32, np.uint8)
            )
        else:
            return None

    def _get_opacity(self, t):
        fade_in = self._abstract.video_fade_in
        fade_out = self._abstract.video_fade_out
        total_audio_dur = self._abstract.total_audio_dur

        if t < fade_in:
            progress = float(t / fade_in)
        elif t < total_audio_dur - fade_out:
            progress = 1.0
        else:
            progress = float((total_audio_dur - t) / fade_out)

        return (1 - cos(progress * pi)) * 0.5

    def _fade_fragment(self, unfaded_fragment, x_slice, opacity):
        noise, fade_float, fade_uint = (
            buf[:, x_slice] for buf in self._fading_buffers
        )

        self._rng.random(dtype=np.float32, out=noise)
        np.multiply(unfaded_frame, opacity, out=fade_float)
        fade_float += 255 - 255 * opacity
        fade_float += noise
        np.minimum(fade_float, 255, out=fade_float)
        fade_uint[:] = fade_float

        return fade_uint

    def _get_faded_fragment(self, unfaded_fragment, out_start_x, opacity):
        if type(unfaded_fragment) is int:
            x_slice = slice(out_start_x, out_start_x + unfaded_fragment)
            return self._blank_frame[:, x_slice]

        if opacity == 1.0:
            return unfaded_fragment

        x_slice = slice(out_start_x, out_start_x + unfaded_fragment.shape[1])
        if opacity == 0.0:
            return self._blank_frame[:, x_slice]
        else:
            return self._fade_fragment(
                unfaded_fragment, x_slice, opacity
            )

    def _get_faded_frame_fragments(
        self, unfaded_frame_fragments_iter, opacity
    ):
        for out_start_x, unfaded_fragment in unfaded_frame_fragments_iter:
            yield self._get_faded_fragment(
                unfaded_fragment, out_start_x, opacity
            )

    def get_fragments(self):
        total_frames = ceil(self._abstract.total_audio_dur * self._format.fps)
        unfaded_fragments = _UnfadedFragments(self._abstract, self._format)

        for frame_num in range(total_frames):
            t = Fraction(frame_num, self._format.fps)

            unfaded_frame_fragments_iter = unfaded_fragments.get_fragments(t)
            opacity = self._get_opacity(t)

            yield self._get_faded_frame_fragments(
                unfaded_frame_fragments_iter, opacity
            )


class ScrollyRender:
    _audio_bitrate = 48_000

    def __init__(
        self, *, scrolly_abstract, scrolly_format, out_path,
        codec_settings=defaults.CODEC_SETTINGS,
    ):
        self._abstract = self._check_abstract(scrolly_abstract)
        self._format = self._check_format(scrolly_format)
        self._out_path = self._sanitise_out_path(out_path)
        self._audio_codec_settings, self._video_codec_settings \
            = self._check_codec_settings(codec_settings)

        self._frames = _Frames(scrolly_abstract, scrolly_format)

    def _check_abstract(self, abstract):
        if isinstance(abstract, ScrollyAbstract):
            return abstract
        else:
            raise TypeError(
                "scrolly_abstract should be a ScrollyAbstract instance"
            )

    def _check_format(self, format_):
        if isinstance(format_, ScrollyFormat):
            return format_
        else:
            raise TypeError(
                "scrolly_format should be a ScrollyFormat instance"
            )

    def _sanitise_out_path(self, out_path):
        out_path = _common.sanitise_path(out_path, "out_path")

        if out_path.suffix == ".mp4":
            return out_path
        else:
            raise ValueError("out_path should end in .mp4")

    def _check_codec_settings(self, codec_settings):
        if (
            type(codec_settings) is tuple
            and isinstance(codec_settings[0], AudioCodecSettings)
            and isinstance(codec_settings[1], VideoCodecSettings)
        ):
            return codec_settings
        else:
            raise TypeError(
                "codec_settings should be a tuple containing an "
                "AudioCodecSettings instance and a VideoCodecSettings "
                "instance"
            )

    def _get_cmd(self):
        pad_before_ms = self._abstract.pad_before * 1000
        return [
            "ffmpeg",

            "-i", self._abstract.audio_path,

            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-r", str(self._format.fps),
            "-s", f"{self._format.out_h}x{self._format.out_w}",
            "-i", "pipe:",

            "-af", f"adelay={pad_before_ms}|{pad_before_ms},apad",
            "-c:a", "aac",
            "-ar", "48000",
            "-ac", "2",
            *self._audio_codec_settings,

            "-vf", "transpose",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            *self._video_codec_settings,

            "-shortest", self._out_path, "-y"
        ]

    def _send_frames_to_proc(self, proc):
        saved_err = None

        try:
            for frame_fragments_iter in self._frames.get_fragments():
                for frame_fragment in frame_fragments_iter:
                    print(frame_fragment.flags["C_CONTIGUOUS"])
                    frame_fragment = np.swapaxes(frame_fragment, 0, 1)
                    print(frame_fragment.flags["C_CONTIGUOUS"])
                    proc.stdin.write(frame_fragment)
        except BaseException as err:
            saved_err = err

        try:
            proc.stdin.close()
            proc.wait()
        except BaseException as err:
            if not saved_err:
                saved_err = err

        if saved_err:
            raise saved_err

    def render(self):
        proc = subprocess.Popen(
            self._get_cmd(),
            stdin=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )

        try:
            self._send_frames_to_proc(proc)
        except BaseException:
            try:
                self._out_path.unlink()
            except BaseException:
                pass

            raise
