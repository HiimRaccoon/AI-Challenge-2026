from wp04.contracts import FrameRecord
from wp04.temporal import LocalTemporalResolver, TemporalResolver


def test_resolver_selects_overlapping_interval_not_only_start():
    frames = [FrameRecord("tv1", "L01_V001", 20, 200, 5500)]
    resolver = LocalTemporalResolver(frames, [("L01_V001", 20, 5000, 6000)])
    assert resolver.frame_hypotheses("L01_V001", 4900, 6100)[0].frame_id == 20


def test_resolver_falls_back_to_nearest_original_frame_at_interval_midpoint():
    frames = [
        FrameRecord("tv1", "L01_V001", 10, 1, 1000),
        FrameRecord("tv1", "L01_V001", 20, 2, 3000),
    ]
    resolver: TemporalResolver = LocalTemporalResolver(frames, [])
    assert resolver.frame_hypotheses("L01_V001", 2300, 2500)[0].frame_id == 20
