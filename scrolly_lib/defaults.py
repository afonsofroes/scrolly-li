from fractions import Fraction

from . import codec_settings


PAD_BEFORE=0
PAD_AFTER=0
VIDEO_FADE_IN=0
VIDEO_FADE_OUT=0
MARGINS_Y_AND_X_IN_OUT_HS = Fraction(1, 25), Fraction(1, 50)

OUT_H_AND_W = 1080, 1920
FPS = 60
DIVISIONS_OF_PIXEL = 5

ADD_BUFFER_COLS = 4_000
CODEC_SETTINGS = codec_settings.TEST_AUDIO, codec_settings.TEST_VIDEO
