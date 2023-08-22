from math import ceil
from fractions import Fraction
import cv2
import subprocess

from . import _common, _motion, defaults


class _PageReader:
    def __init__(self, path, margins_y_and_x_in_out_hs):
        self._path = path
        self._margins_y_in_out_hs, self._margins_x_in_out_hs \
            = margins_y_and_x_in_out_hs

        self._untrimmed_img = None
        self._load_untrimmed_img()
        self._untrimmed_h, self._untrimmed_w, *_ = self._untrimmed_img.shape
        self._trim_slices, self.w_as_mul_of_h = self._get_trim_slices()
        self._unload_untrimmed_img()

    def _load_untrimmed_img(self):
        self._untrimmed_img = cv2.imread(str(self._path))

    def _unload_untrimmed_img(self):
        self._untrimmed_img = None

    def _get_body_positions(self):
        start_y = 0
        for y in range(self._untrimmed_h):
            if self._untrimmed_img[y:y+1].min() < 230:
                start_y += 1
            else:
                break
        else:  # if no break
            raise Exception(f"page at {self._path} considered empty")

        stop_y = self._untrimmed_h
        for y in range(self._untrimmed_h, 0, -1):
            if self._untrimmed_img[y-1:y].min() < 230:
                stop_y -= 1
            else:
                break

        view = self._untrimmed_img[start_y:stop_y]

        start_x = 0
        for x in range(self._untrimmed_w):
            if view[:, x:x+1].min() < 230:
                start_x += 1
            else:
                break

        stop_x = self._untrimmed_w
        for x in range(self._untrimmed_w, 0, -1):
            if view[:, x-1:x].min() < 230:
                stop_x -= 1
            else:
                break

        return start_y, stop_y, start_x, stop_x

    def _get_trim_slice_y(self, body_start_y, body_stop_y):
        ideal_margin_proportion_y = self._margins_y_in_out_hs * 2
        ideal_body_proportion_y = 1 - ideal_margin_proportion_y
        ideal_margin_in_bodies_y = (
            ideal_margin_proportion_y / ideal_body_proportion_y
        )

        body_h = body_stop_y - body_start_y
        ideal_margin_y = round(body_h * ideal_margin_in_bodies_y)

        top_margin_too_short = body_start_y < ideal_margin_y
        bottom_margin_too_short = (
            (self._untrimmed_h - body_stop_y) < ideal_margin_y
        )

        match top_margin_too_short, bottom_margin_too_short:
            case True, True:
                trim_start_y, trim_stop_y = 0, self._untrimmed_h
            case (True, False) | (False, True):
                mul_of_body_and_margin_for_margin = (
                    ideal_margin_proportion_y
                    / (1 - ideal_margin_proportion_y)
                )

                if top_margin_too_short:
                    trim_start_y = 0

                    bottom_margin = round(
                        body_stop_y * mul_of_body_and_margin_for_margin
                    )
                    trim_stop_y = body_stop_y + bottom_margin
                else:
                    top_margin = round(
                        (self._untrimmed_h - body_start_y)
                        * mul_of_body_and_margin_for_margin
                    )
                    trim_start_y = body_start_y - top_margin

                    trim_stop_y = self._untrimmed_h
            case False, False:
                trim_start_y = body_start_y - ideal_margin_y
                trim_stop_y = body_stop_y + ideal_margin_y

        trim_slice_y = slice(trim_start_y, trim_stop_y)
        trimmed_h = trim_stop_y - trim_start_y
        return trim_slice_y, trimmed_h

    def _get_trim_slice_x(self, body_start_x, body_stop_x, trimmed_h):
        ideal_margin_x = round(trimmed_h * self._margins_x_in_out_hs)

        trim_start_x = max(body_start_x - ideal_margin_x, 0)
        trim_stop_x = min(body_stop_x + ideal_margin_x, self._untrimmed_w)

        trim_slice_x = slice(trim_start_x, body_stop_x)
        trimmed_w = body_stop_x - body_start_x
        return trim_slice_x, trimmed_w

    def _get_trim_slices(self):
        body_start_y, body_stop_y, body_start_x, body_stop_x \
            = self._get_body_positions()

        trim_slice_y, trimmed_h = self._get_trim_slice_y(
            body_start_y, body_stop_y
        )
        trim_slice_x, trimmed_w = self._get_trim_slice_x(
            body_start_x, body_stop_x, trimmed_h
        )

        return (
            (trim_slice_y, trim_slice_x),
            Fraction(trimmed_w, trimmed_h)
        )

    def get_trimmed_img(self):
        self._load_untrimmed_img()
        trimmed_img = self._untrimmed_img[self._trim_slices]
        self._unload_untrimmed_img()
        return trimmed_img


class _Page:
    def __init__(self, path, margins_y_and_x_in_out_hs, in_t, out_t):
        self._reader = _PageReader(path, margins_y_and_x_in_out_hs)
        self._in_t, self._out_t = in_t, out_t

        self.dur = out_t - in_t
        self.w_as_mul_of_h = self._reader.w_as_mul_of_h

        self._page_before = self._page_after = None
        self._motion_calculator = None

    def set_page_before(self, page_before):
        self._page_before = page_before

    def set_page_after(self, page_after):
        self._page_after = page_after

    def set_motion_calculator(self, motion_calculator):
        self._motion_calculator = motion_calculator

    def get_trimmed_img(self):
        return self._reader.get_trimmed_img()

    def check_t(self, t):
        return self._in_t <= t < self._out_t

    def get_x_offset_as_mul_of_h(self, t):
        return self._motion_calculator.get_x(t - self._in_t)


class ScrollyAbstract:
    def __init__(
        self, *, audio_path, timestamps_and_page_paths,
        pad_before=defaults.PAD_BEFORE,
        pad_after=defaults.PAD_AFTER,
        video_fade_in=defaults.VIDEO_FADE_IN,
        video_fade_out=defaults.VIDEO_FADE_OUT,
        margins_y_and_x_in_out_hs=defaults.MARGINS_Y_AND_X_IN_OUT_HS
    ):
        self.audio_path = _common.sanitise_path(audio_path, "audio_path")
        self.pad_before = self._sanitise_pad_before(pad_before)
        self._pad_after = _common.sanitise_time(pad_after, "pad_after")

        self._raw_audio_dur = self._get_raw_audio_dur()
        self.total_audio_dur = self._get_total_audio_dur()

        self.video_fade_in, self.video_fade_out \
            = self._sanitise_video_fades(video_fade_in, video_fade_out)
        self._margins_y_and_x_in_out_hs \
            = tuple(self._sanitise_margins_y_and_x_in_out_hs(
                  margins_y_and_x_in_out_hs
              ))

        self.pages = list(self._get_pages(timestamps_and_page_paths))
        self._set_pages_before_and_after()
        self._in_vel, self._out_vel = self._set_pages_motion_info_get_vels()

    def _sanitise_pad_before(self, pad_before):
        pad_before = _common.sanitise_time(pad_before, "pad_before")
        pad_before_ms = ceil(1000 * pad_before)
        return Fraction(pad_before_ms, 1000)

    def _get_raw_audio_dur(self):
        ffprobe_str = subprocess.check_output(
            ["ffprobe", "-i", self.audio_path],
            stderr=subprocess.STDOUT
        ).decode()
        reported_dur_str = ffprobe_str.split("Duration: ")[1].split(", ")[0]
        reported_dur = _common.sanitise_time(
            reported_dur_str, "audio duration as reported by ffprobe"
        )

        return reported_dur + Fraction(1, 100)

    def _get_total_audio_dur(self):
        return self.pad_before + self._raw_audio_dur + self._pad_after

    def _sanitise_to_tapp_list(self, timestamps_and_page_paths):
        try:
            tapp_iter = iter(timestamps_and_page_paths)
        except TypeError:
            raise TypeError("timestamps_and_page_paths must be iterable")

        tapp = list(tapp_iter)
        if len(tapp) % 2 == 0:
            raise ValueError(
                "timestamps_and_page_paths should contain an odd number of "
                "items"
            )
        elif len(tapp) == 1:
            raise ValueError("not enough items in timestamps_and_page_paths")
        else:
            return tapp

    def _sanitise_video_fades(self, video_fade_in, video_fade_out):
        video_fade_in = _common.sanitise_time(
            video_fade_in, "video_fade_in"
        )
        video_fade_out = _common.sanitise_time(
            video_fade_out, "video_fade_out"
        )

        if video_fade_in + video_fade_out > self.total_audio_dur:
            raise ValueError(
                "video_fade_in + video_fade_out is greater than the total "
                "audio duration"
            )

        yield video_fade_in
        yield video_fade_out

    def _sanitise_margins_y_and_x_in_out_hs(
        self, margins_y_and_x_in_out_hs
    ):
        err_val = (
            "margins_y_and_x_in_body_hs should be a tuple containing 2 "
            "Fractions or floats in range 0 <= x < 0.5"
        )

        if (
            type(margins_y_and_x_in_out_hs) is not tuple
            or len(margins_y_and_x_in_out_hs) != 2
        ):
            raise TypeError(err_val)

        for margin_in_out_hs in margins_y_and_x_in_out_hs:
            if not isinstance(margin_in_out_hs, Fraction | float):
                raise TypeError(err_val)

            if not 0 <= margin_in_out_hs < 0.5:
                raise ValueError(err_val)

            yield margin_in_out_hs

    def _get_pages(self, timestamps_and_page_paths):
        tapp = self._sanitise_to_tapp_list(timestamps_and_page_paths)

        unpadded_in_t = _common.sanitise_time(
            tapp[0], "timestamps_and_page_paths[0]"
        )
        if unpadded_in_t < 0:
            raise ValueError(
                "timestamps in timestamps_and_page_paths should not be "
                "negative"
            )

        for page_i, (path, unpadded_out_t_str) in enumerate(
            zip(tapp[1::2], tapp[2::2])
        ):
            path = _common.sanitise_path(
                path, f"timestamps_and_page_paths[{page_i * 2 + 1}]"
            )
            unpadded_out_t = _common.sanitise_time(
                unpadded_out_t_str,
                f"timestamps_and_page_paths[{page_i * 2 + 2}]"
            )

            if unpadded_out_t <= unpadded_in_t:
                raise ValueError(
                    "timestamps in timestamps_and_page_paths should keep "
                    "increasing"
                )

            in_t, out_t = self.pad_before + in_t, self.pad_before + out_t
            yield _Page(
                path, self._margins_y_and_x_in_out_hs,
                in_t, out_t
            )

            unpadded_in_t = unpadded_out_t

        if unpadded_out_t > self._raw_audio_dur:
            raise ValueError(
                "timestamps in timestamps_and_page_paths should not exceed "
                "the duration of the audio file"
            )

    def _set_pages_before_and_after(self):
        for before, after in zip(self.pages[:-1], self.pages[1:]):
            before.set_page_after(after)
            after.set_page_before(before)

    def _set_pages_motion_info_get_vels(self):
        durations = np.asarray(
            (page.dur for page in self.pages), dtype=np.float32
        )
        widths = np.asarray(
            (page.w_as_mul_of_h for page in self.pages), dtype=np.float32
        )
        page_motion_calculators, in_vel, out_vel = _motion.calculate(
            durations, widths
        )

        for page, calculator in zip(self.pages, page_motion_calculators):
            page.set_motion_calculator(calculator)

        return in_vel, out_vel

    def check_page_i(self, page_i, t):
        if 0 <= page_i < len(self.pages):
            return self.pages[page_i].check_t(t)
        elif page_i == -1:
            return t < self.pad_before
        else:
            return t >= self.total_audio_dur - self._pad_after

    def get_x_before_as_mul_of_h(self, t):
        time_before = self.pad_before - t
        return self._in_vel * time_before

    def get_x_after_as_mul_of_h(self, t):
        time_after = t - self._pad_after
        self._out_vel * time_after
