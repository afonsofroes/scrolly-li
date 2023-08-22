from ._abstract import ScrollyAbstract
from ._format import ScrollyFormat
from ._render import ScrollyRender
from . import defaults


def make_scrolly(
    audio_path, timestamps_and_page_paths, out_path, *,

    pad_before=defaults.PAD_BEFORE,
    pad_after=defaults.PAD_AFTER,
    video_fade_in=defaults.VIDEO_FADE_IN,
    video_fade_out=defaults.VIDEO_FADE_OUT,
    margins_y_and_x_in_out_hs=defaults.MARGINS_Y_AND_X_IN_OUT_HS,

    out_h_and_w=defaults.OUT_H_AND_W,
    fps=defaults.FPS,
    divisions_of_pixel=defaults.DIVISIONS_OF_PIXEL,

    codec_settings=defaults.CODEC_SETTINGS,
):
    abstract = ScrollyAbstract(
        audio_path=audio_path,
        timestamps_and_page_paths=timestamps_and_page_paths,
        pad_before=pad_before,
        pad_after=pad_after,
        video_fade_in=video_fade_in,
        video_fade_out=video_fade_out,
        margins_y_and_x_in_out_hs=margins_y_and_x_in_out_hs
    )
    format_ = ScrollyFormat(
        out_h_and_w=out_h_and_w,
        fps=fps,
        divisions_of_pixel=divisions_of_pixel
    )
    render = ScrollyRender(
        scrolly_abstract=abstract,
        scrolly_format=format_,
        out_path=out_path,
        codec_settings=codec_settings
    )
    render.render()
