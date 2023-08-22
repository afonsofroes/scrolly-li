import random
import numpy as np


_MIN_VEL_AS_MUL_OF_AVG_VEL = 0.1
_HISTORY_LEN = 10


random.seed(0)


class _Segment:
    def __init__(self, start_t, stop_t, a, b, c):
        self.start_t, self.stop_t = start_t, stop_t
        self.a, self.b, self.c = a, b, c


class _PageCalculator:
    def __init__(self, dur, w, in_vel, out_vel):
        self._dur, self._w = dur, w

        self._avg_vel = w / dur
        self._min_vel = self._avg_vel * _MIN_VEL_AS_MUL_OF_AVG_VEL

        self._in_vel = max(in_vel, self._min_vel)
        self._out_vel = max(out_vel, self._min_vel)

        self._segments = self._calculate_segments()
        self.max_abs_accel = max(abs(segment.a) for segment in self._segments)

    def _calculate_2_segments(self):
        avg_vel_without_shift = (self._in_vel + self._out_vel) / 2
        vel_increase_at_shift = 2 * (self._avg_vel - avg_vel_without_shift)
        vel_increase_is_positive = vel_increase_at_shift >= 0

        default_accel = (self._out_vel - self._in_vel) / self._dur

        shift_t = self._dur / 2
        prev_adjustment = 0.0
        adjustment = shift_t / 2
        while adjustment != prev_adjustment:
            unadjusted_shift_vel = self._in_vel + shift_t * default_accel
            shift_vel = unadjusted_shift_vel + vel_increase_at_shift
            in_accel = (shift_vel - self._in_vel) / shift_t
            out_accel = (self._out_vel - shift_vel) / (self._dur - shift_t)

            if vel_increase_is_positive:
                should_delay_shift = in_accel + out_accel >= 0
            else:
                should_delay_shift = in_accel + out_accel < 0

            if should_delay_shift:
                shift_t += adjustment
            else:
                shift_t -= adjustment

            prev_adjustment = adjustment
            adjustment *= 0.5

        shift_x = (self._in_vel + shift_vel) / 2 * shift_t

        return [
            _Segment(
                start_t=0.0, stop_t=shift_t,
                a=in_accel, b=self._in_vel, c=0.0
            ),
            _Segment(
                start_t=shift_t, stop_t=self._dur,
                a=out_accel, b=shift_vel, c=shift_x
            )
        ]

    def _calculate_3_segments(self):
        in_above_min_vel = self._in_vel - self._min_vel
        out_above_min_vel = self._out_vel - self._min_vel

        w_remaining_after_min_vel = self._w - self._min_vel * self._dur
        w_covered_by_abs_accel_1 = (
            (in_above_min_vel ** 2 + out_above_min_vel ** 2) / 2
        )
        abs_accel = w_covered_by_abs_accel_1 / w_remaining_after_min_vel

        in_dur = in_above_min_vel / abs_accel
        out_dur = out_above_min_vel / abs_accel
        in_w = (self._in_vel + self._min_vel) / 2 * in_dur
        out_w = (self._min_vel + self._out_vel) / 2 * out_dur

        return [
            _Segment(
                start_t=0.0, stop_t=in_dur,
                a=-abs_accel, b=self._in_vel, c=0.0
            ),
            _Segment(
                start_t=in_dur, stop_t=self._dur - out_dur,
                a=0.0, b=self._min_vel, c=in_w
            ),
            _Segment(
                start_t=self._dur - out_dur, stop_t=self._dur,
                a=abs_accel, b=self._min_vel, c=self._w - out_w
            )
        ]

    def _calculate_segments(self):
        segments = self._calculate_2_segments()
        if segments[1].b >= self._min_vel:
            return segments
        else:
            return self._calculate_3_segments()

    def _get_segment(self, t):
        for segment in self._segments:
            if segment.start_t <= t < segment.stop_t:
                break

        return segment

    def get_x(self, t):
        segment = self._get_segment(t)
        t -= segment.start_t
        a, b, c = segment.a, segment.b, segment.c
        return 0.5*a*t*t + b*t + c


class _BorderVelScorer:
    def __init__(self, durations, widths):
        self._durations, self._widths = durations, widths
        self._min_vels_for_borders = self._get_min_vels_for_borders()

    def _get_min_vels_for_borders(self):
        avg_vels_for_pages = self._widths / self._durations
        min_vels_for_pages = avg_vels_for_pages * _MIN_VEL_AS_MUL_OF_AVG_VEL

        min_vels_for_borders = np.empty(
            self._widths.shape[0] + 1, dtype=np.float32
        )
        min_vels_for_borders[0] = min_vels_for_pages[0]
        min_vels_for_borders[-1] = min_vels_for_pages[-1]
        np.maximum(
            min_vels_for_pages[:-1], min_vels_for_pages[1:],
            out=min_vels_for_borders[1:-1]
        )

        return min_vels_for_borders

    def _get_max_accels(self, attempt):
        return [
            _PageCalculator(*args).max_abs_accel
            for args in zip(
                self._durations, self._widths, attempt[:-1], attempt[1:]
            )
        ]

    def get_scores(self, attempt):
        attempt_with_floor = np.maximum(attempt, self._min_vels_for_borders)
        scores = np.abs(attempt - attempt_with_floor)

        max_accels = self._get_max_accels(attempt_with_floor)

        for border_vel_i in range(attempt.shape[0]):
            for offset, score_mul in [
                (-3, 0.5), (-2, 0.7), (-1, 1.0),
                (0, 1.0), (1, 0.7), (2, 0.5)
            ]:
                page_i = border_vel_i + offset
                if 0 <= page_i < len(max_accels):
                    scores[border_vel_i] += max_accels[page_i] * score_mul

        return scores


def _hone_in(length, scorer):
    count = 500

    add_range = 0.001, 0.0012

    candidates_history = np.zeros((_HISTORY_LEN, length), dtype=np.float32)
    scores_history = np.empty((_HISTORY_LEN, length), dtype=np.float32)

    for i, candidate in enumerate(candidates_history):
        scores_history[i] = scorer.get_scores(candidate)

    balanced_weightings = [
        (_HISTORY_LEN - 1) / 2 - x
        for x in range(_HISTORY_LEN)
    ]
    abs_sum_balanced_weightings = sum(abs(x) for x in balanced_weightings)
    scaled_weightings = [
        x / abs_sum_balanced_weightings
        for x in balanced_weightings
    ]

    for _ in range(count):
        try_new_vals = np.full(length, False)
        for i in range(len(try_new_vals)):
            try_new_vals[i] = random.random() < 1

        new_candidates = candidates_history[-1].copy()

        for i, try_new_val in enumerate(try_new_vals):
            if not try_new_val:
                continue

            candidates_and_scores = list(
                zip(candidates_history[:, i], scores_history[:, i])
            )
            candidates_and_scores.sort(key=lambda x: x[1])

            avg_candidate = np.mean(candidates_history[:, i])

            candidate_offsets = [
                candidate - avg_candidate
                for candidate, _ in candidates_and_scores
            ]

            new_offset = (
                (random.random() * 2 - 1)
                * (add_range[1] / add_range[0]) ** random.random()
                * add_range[0]
            )
            for offset, weight in zip(candidate_offsets, scaled_weightings):
                new_offset += offset * weight

            new_candidates[i] = avg_candidate + new_offset

        new_scores = scorer.get_scores(new_candidates)

        for i, try_new_val in enumerate(try_new_vals):
            if not try_new_val:
                continue

            if new_scores[i] > np.mean(scores_history[:, i]):
                continue

            candidates_history[:-1, i] = candidates_history[1:, i]
            candidates_history[-1, i] = new_candidates[i]
            scores_history[:-1, i] = scores_history[1:, i]
            scores_history[-1, i] = new_scores[i]

        print(new_candidates)
        print(new_scores)
        print('')
        yield candidates_history[-1]


def _get_page_calculators(durations, widths, border_vels):
    return [
        _PageCalculator(dur, w, in_vel, out_vel)
        for dur, w, in_vel, out_vel
        in zip(durations, widths, border_vels[:-1], border_vels[1:])
    ]


def _display(durations, widths, page_calculators):
    import cv2

    h_and_w = 400, 1000
    canvas = np.full((h_and_w), 255, dtype=np.uint8)

    end_times = np.cumsum(durations)
    start_times = end_times - durations
    dur = end_times[-1]

    end_xs = np.cumsum(widths)
    start_xs = end_xs - widths
    w = end_xs[-1]

    t = 0
    for start_t, end_t, start_x, page_calculator in zip(
        start_times, end_times, start_xs, page_calculators
    ):
        arr_x_pos = int(min(t / dur * h_and_w[1], h_and_w[1]-1))
        canvas[:, arr_x_pos] = 128

        x = start_x + page_calculator.get_x(0)
        arr_y_pos = int(min(x / w * h_and_w[0], h_and_w[0]-1))
        canvas[-1-arr_y_pos] = 128

        while t < end_t:
            arr_x_pos = int(min(t / dur * h_and_w[1], h_and_w[1]-1))

            x = start_x + page_calculator.get_x(t - start_t)
            arr_y_pos = int(min(x / w * h_and_w[0], h_and_w[0]-1))

            canvas[-1-arr_y_pos, arr_x_pos] = 0

            t += dur / h_and_w[1] / 10

    cv2.imshow('', canvas)
    cv2.waitKey(1)


def calculate(durations, widths):
    scorer = _BorderVelScorer(durations, widths)
    for border_vels in _hone_in(len(durations) + 1, scorer, 1000):
        page_calculators = _get_page_calculators(
            durations, widths, border_vels
        )
        _display(durations, widths, page_calculators)

    return page_calculators, border_vels[0], border_vels[-1]
