"""Compact Arrow-backed access to immutable embedding metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .contracts import ContractError


_REQUIRED_COLUMNS = ("video_id", "frame_id", "timestamp_ms", "keyframe_path")


@dataclass(frozen=True)
class SearchMapping:
    """Expose row-like access without materializing one Python dict per vector."""

    table: pa.Table

    def __len__(self) -> int:
        return self.table.num_rows

    @property
    def has_shot_ids(self) -> bool:
        return "shot_id" in self.table.column_names

    def __getitem__(self, vector_id: int) -> dict[str, object]:
        if vector_id < 0 or vector_id >= len(self):
            raise IndexError(vector_id)
        return {
            name: self.table.column(name)[vector_id].as_py()
            for name in self.table.column_names
        }

    def shot_neighbor_ids(self, vector_id: int, count: int) -> tuple[int, ...]:
        """Return nearby vector IDs that belong to the same contiguous TV1 shot."""
        if count <= 0 or not self.has_shot_ids:
            return ()
        video_column = self.table.column("video_id")
        shot_column = self.table.column("shot_id")
        video_id = video_column[vector_id].as_py()
        shot_id = shot_column[vector_id].as_py()
        neighbors: list[int] = []
        left, right = vector_id - 1, vector_id + 1
        left_open = right_open = True
        while len(neighbors) < count and (left_open or right_open):
            if left_open:
                if (
                    left >= 0
                    and video_column[left].as_py() == video_id
                    and shot_column[left].as_py() == shot_id
                ):
                    neighbors.append(left)
                    left -= 1
                else:
                    left_open = False
            if len(neighbors) >= count:
                break
            if right_open:
                if (
                    right < len(self)
                    and video_column[right].as_py() == video_id
                    and shot_column[right].as_py() == shot_id
                ):
                    neighbors.append(right)
                    right += 1
                else:
                    right_open = False
        return tuple(neighbors)

    def vector_ids_for_frames(
        self, frames: Sequence[Mapping[str, object]],
    ) -> tuple[int, ...]:
        """Resolve an explicit TV1 frame scope to vectors without guessing time.

        The mapping remains Arrow-backed.  We first narrow the table with an
        Arrow ``is_in`` operation on video IDs, then inspect only those rows in
        Python.  This avoids materialising a 700k-entry identity dictionary for
        every resident visual model while keeping the scope lookup cheap.
        """
        wanted: set[tuple[str, int]] = set()
        for frame in frames:
            video_id, frame_id = frame.get("video_id"), frame.get("frame_id")
            if (
                not isinstance(video_id, str) or not video_id
                or isinstance(frame_id, bool) or not isinstance(frame_id, int)
                or frame_id < 0
            ):
                raise ContractError("candidate_frames require video_id and non-negative frame_id")
            wanted.add((video_id, frame_id))
        if not wanted:
            return ()
        video_ids = sorted({video_id for video_id, _ in wanted})
        mask = pc.is_in(
            self.table.column("video_id"), value_set=pa.array(video_ids, type=pa.string()),
        )
        candidate_ids = pc.indices_nonzero(mask).to_pylist()
        video_column = self.table.column("video_id")
        frame_column = self.table.column("frame_id")
        return tuple(
            int(vector_id) for vector_id in candidate_ids
            if (
                str(video_column[vector_id].as_py()),
                int(frame_column[vector_id].as_py()),
            ) in wanted
        )


def load_search_mapping(path: Path) -> SearchMapping:
    try:
        schema = pq.read_schema(path)
        missing = set(_REQUIRED_COLUMNS) - set(schema.names)
        if missing:
            raise ContractError(
                f"embedding map lacks required columns: {', '.join(sorted(missing))}"
            )
        columns = list(_REQUIRED_COLUMNS)
        for optional in ("vector_id", "shot_id"):
            if optional in schema.names:
                columns.append(optional)
        return SearchMapping(pq.read_table(path, columns=columns))
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise ContractError("embedding map cannot be read") from exc
