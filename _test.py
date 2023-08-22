import numpy as np

from scrolly_lib import _motion


durations = np.asarray(
    [10, 60, 15, 120, 40, 5, 30],
    dtype=np.float32
)
widths = np.asarray(
    [0.7, 0.6, 0.7, 0.75, 0.75, 0.6, 0.6],
    dtype=np.float32
)


_motion.calculate(durations*1.2, widths*1.2)
