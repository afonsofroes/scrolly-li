from pathlib import Path
from fractions import Fraction


def sanitise_path(path, name):
    try:
        return Path(path)
    except TypeError:
        raise TypeError(
            f"failed to convert {name} {path!r} to pathlib.Path"
        ) from None


def sanitise_time(time, name):
    if isinstance(time, Fraction | int | float):
        time = Fraction(time)
    elif isinstance(time, str):
        units = time.split(':')
        if len(units) <= 3:
            time = Fraction(0)
            for unit in units:
                time *= 60
                time += Fraction(unit)
        else:
            raise ValueError(
                f"{name} should have a maximum of 3 colon-separated units "
                "(h:m:s)"
            )
    else:
        raise TypeError(f"{name} should be a Fraction, int, float, or str")

    if time < 0:
        raise ValueError(f"{name} should not be less than 0")

    return time
