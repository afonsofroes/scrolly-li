class AudioCodecSettings(list):
    def __init__(self, *xargs, _instantiate=False, **_):
        if _instantiate:
            super().__init__(xargs[0])
        else:
            raise TypeError(
                f"{type(self).__name__} should be instantiated using the "
                ".from_bitrate method"
            )

    @classmethod
    def from_bitrate(cls, bitrate):
        if type(bitrate) is not int:
            raise TypeError("bitrate should be an int")

        if bitrate <= 0:
            raise ValueError("bitrate should be greater than 0")

        return cls(["-b:a", str(bitrate)], _instantiate=True)


class VideoCodecSettings(list):
    def __init__(self, *xargs, _instantiate=False, **_):
        if _instantiate:
            super().__init__(xargs[0])
        else:
            raise TypeError(
                f"{type(self).__name__} should be instantiated using the "
                ".from_crf_tune_preset method"
            )

    @classmethod
    def from_crf_tune_preset(cls, crf, tune, preset):
        if type(crf) is not int:
            raise TypeError("crf should be an int")

        if not 1 <= crf <= 51:
            raise ValueError("crf should be in range 1 <= x <= 51")

        if type(tune) is not str:
            raise TypeError("tune should be a str")

        tune_vals = [
            "film", "animation", "grain", "stillimage", "fastdecode",
            "zerolatency"
        ]
        if tune not in tune_vals:
            raise ValueError(
                "tune should be one of these values: "
                + ', '.join(map(repr, tune_vals))
            )

        if type(preset) is not str:
            raise TypeError("preset should be a str")

        preset_vals = [
            "ultrafast", "superfast", "veryfast", "faster", "fast", "medium",
            "slow", "slower", "veryslow"
        ]
        if preset not in preset_vals:
            raise ValueError(
                "preset should be one of these values: "
                + ', '.join(map(repr, preset_vals))
            )

        return cls(
            ["-crf", str(crf), "-tune", tune, "-preset", preset],
            _instantiate=True
        )


TEST_AUDIO = AudioCodecSettings.from_bitrate(60_000)
TEST_VIDEO = VideoCodecSettings.from_crf_tune_preset(
    42, "animation", "superfast"
)
