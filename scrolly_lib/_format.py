from fractions import Fraction

from . import defaults


class ScrollyFormat:
    def __init__(
        self, *,
        out_h_and_w=defaults.OUT_H_AND_W,
        fps=defaults.FPS,
        divisions_of_pixel=defaults.DIVISIONS_OF_PIXEL
    ):
        self.out_h, self.out_w = self._check_out_h_and_w(out_h_and_w)
        self.fps = self._sanitise_fps(fps)
        self.divisions_of_pixel = self._check_divisions_of_pixel(
            divisions_of_pixel
        )

        self.out_shape = self.out_h, self.out_w, 3
        self.frame_dur = 1 / self.fps

    def _check_out_h_and_w(self, out_h_and_w):
        err = "out_h_and_w should be a tuple containing 2 ints greater than 0"

        if (
            type(out_h_and_w) is not tuple
            or len(out_h_and_w) != 2
        ):
            raise TypeError(err)

        for d in out_h_and_w:
            if type(d) is not int:
                raise TypeError(err)

            if d <= 0:
                raise ValueError(err)

        return out_h_and_w

    def _sanitise_fps(self, fps):
        if not isinstance(fps, Fraction | int | float | str):
            raise TypeError("time should be a Fraction, int, float, or str")

        fps = Fraction(fps)

        if fps <= 0:
            raise ValueError("fps should be greater than 0")

        return fps

    def _check_divisions_of_pixel(self, divisions_of_pixel):
        if type(divisions_of_pixel) is not int:
            raise TypeError("divisions of pixel should be an int")

        if divisions_of_pixel <= 0:
            raise ValueError("divisions_of_pixel should be greater than 0")

        return divisions_of_pixel
