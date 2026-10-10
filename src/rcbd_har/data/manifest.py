from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {"path", "participant", "label"}
BILATERAL_PATH_COLUMNS = ("left_path", "right_path")
BILATERAL_VALIDITY_COLUMNS = ("left_validity_path", "right_validity_path")
FOUR_LIMB_PATH_COLUMNS = (
    "left_wrist_path",
    "right_wrist_path",
    "left_ankle_path",
    "right_ankle_path",
)
FOUR_LIMB_VALIDITY_COLUMNS = tuple(
    f"{column.removesuffix('_path')}_validity_path" for column in FOUR_LIMB_PATH_COLUMNS
)
ANCHOR_PATH_COLUMN = "anchor_path"
ANCHOR_VALIDITY_COLUMN = "anchor_validity_path"


def load_manifest(path: str | Path, require_labels: bool = True) -> pd.DataFrame:
    path = Path(path).resolve()
    frame = pd.read_csv(path)
    bilateral = "path" in frame.columns
    continuous_bilateral = set(BILATERAL_PATH_COLUMNS).issubset(frame.columns)
    four_limb = set(FOUR_LIMB_PATH_COLUMNS).issubset(frame.columns)
    required = {"participant"} | ({"label"} if require_labels else set())
    missing = required.difference(frame.columns)
    if missing or not (bilateral or continuous_bilateral or four_limb):
        expected = (
            "path, left_path+right_path, or all four limb *_path columns, "
            "plus participant and optional label"
        )
        raise ValueError(f"Manifest lacks required columns ({expected}): {path}")
    if frame.empty:
        raise ValueError(f"Manifest is empty: {path}")
    frame = frame.copy()
    frame["participant"] = frame["participant"].astype(str)
    if "label" not in frame:
        frame["label"] = -1
    frame["label"] = frame["label"].astype(int)
    root = path.parent
    path_columns = [
        column for column in frame.columns if column == "path" or column.endswith("_path")
    ]
    for column in path_columns:
        frame[column] = frame[column].map(
            lambda p: str((root / str(p)).resolve())
            if not Path(str(p)).is_absolute()
            else str(p)
        )
    return frame


def _window_zscore(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    return (array - mean) / np.maximum(std, eps)


def repair_nonfinite_arrays(
    arrays: dict[str, np.ndarray], strategy: str
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Repair missing samples per window and report original missing fractions."""
    repaired = {key: np.asarray(value, dtype=np.float32).copy() for key, value in arrays.items()}
    fractions = {key: float((~np.isfinite(value)).mean()) for key, value in repaired.items()}
    if not any(fractions.values()):
        return repaired, fractions
    if strategy == "error":
        raise ValueError(f"Non-finite sensor samples encountered: {fractions}")
    if strategy not in {"interpolate", "contralateral"}:
        raise ValueError(f"Unknown missing_strategy: {strategy}")
    if strategy == "contralateral":
        pairs = (("left_wrist", "right_wrist"), ("left_ankle", "right_ankle"), ("left", "right"))
        for left_key, right_key in pairs:
            if left_key not in repaired or right_key not in repaired:
                continue
            left_original, right_original = repaired[left_key].copy(), repaired[right_key].copy()
            left_missing = ~np.isfinite(repaired[left_key])
            right_missing = ~np.isfinite(repaired[right_key])
            repaired[left_key][left_missing & np.isfinite(right_original)] = right_original[
                left_missing & np.isfinite(right_original)
            ]
            repaired[right_key][right_missing & np.isfinite(left_original)] = left_original[
                right_missing & np.isfinite(left_original)
            ]
    for key, values in repaired.items():
        for channel in range(values.shape[1]):
            signal = values[:, channel]
            finite = np.isfinite(signal)
            if finite.all():
                continue
            if finite.any():
                positions = np.arange(len(signal))
                signal[~finite] = np.interp(positions[~finite], positions[finite], signal[finite])
            else:
                signal[:] = 0.0
        if not np.isfinite(values).all():
            raise RuntimeError(f"Missing-value repair failed for {key}")
    return repaired, fractions


def _iter_training_arrays(
    frame: pd.DataFrame, limb_mode: str
) -> Iterator[tuple[str, np.ndarray, np.ndarray | None]]:
    if limb_mode == "four" and set(FOUR_LIMB_PATH_COLUMNS).issubset(frame.columns):
        columns = list(FOUR_LIMB_PATH_COLUMNS)
        if ANCHOR_PATH_COLUMN in frame.columns:
            columns.append(ANCHOR_PATH_COLUMN)
        for column in columns:
            key = column.removesuffix("_path")
            validity_column = f"{key}_validity_path"
            for path in frame[column].drop_duplicates():
                source_rows = frame[frame[column] == path]
                validity = None
                if validity_column in frame:
                    validity_paths = source_rows[validity_column].drop_duplicates().tolist()
                    if len(validity_paths) != 1:
                        raise ValueError(f"Sensor source {path} maps to multiple validity arrays")
                    validity = np.load(
                        str(validity_paths[0]), mmap_mode="r", allow_pickle=False
                    )
                yield key, np.load(str(path), mmap_mode="r", allow_pickle=False), validity
        return
    if limb_mode == "bilateral" and set(BILATERAL_PATH_COLUMNS).issubset(frame.columns):
        for column in BILATERAL_PATH_COLUMNS:
            key = column.removesuffix("_path")
            validity_column = f"{key}_validity_path"
            for path in frame[column].drop_duplicates():
                source_rows = frame[frame[column] == path]
                validity = None
                if validity_column in frame:
                    validity_paths = source_rows[validity_column].drop_duplicates().tolist()
                    if len(validity_paths) != 1:
                        raise ValueError(f"Sensor source {path} maps to multiple validity arrays")
                    validity = np.load(
                        str(validity_paths[0]), mmap_mode="r", allow_pickle=False
                    )
                yield key, np.load(str(path), mmap_mode="r", allow_pickle=False), validity
        return
    for path_value in frame["path"].drop_duplicates():
        with np.load(str(path_value), allow_pickle=False) as sample:
            keys = (
                ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
                if limb_mode == "four"
                else ("left", "right")
            )
            if limb_mode == "four" and "anchor" in sample.files:
                keys = (*keys, "anchor")
            for key in keys:
                yield key, np.asarray(sample[key], dtype=np.float32), None


def estimate_global_normalization(
    frame: pd.DataFrame,
    limb_mode: str,
    mode: str = "train_global",
    chunk_size: int = 1_000_000,
) -> dict[str, dict[str, list[float]]]:
    """Estimate per-limb/channel statistics from training sources only."""
    source_columns = (
        [
            *FOUR_LIMB_PATH_COLUMNS,
            *([ANCHOR_PATH_COLUMN] if ANCHOR_PATH_COLUMN in frame.columns else []),
        ]
        if limb_mode == "four" and set(FOUR_LIMB_PATH_COLUMNS).issubset(frame.columns)
        else (
            list(BILATERAL_PATH_COLUMNS)
            if limb_mode == "bilateral"
            and set(BILATERAL_PATH_COLUMNS).issubset(frame.columns)
            else ["path"]
        )
    )
    for column in source_columns:
        ownership = frame.groupby(column)["participant"].nunique()
        if (ownership > 1).any():
            raise ValueError(f"A {column} source is shared across participants")
    totals: dict[str, dict[str, np.ndarray]] = {}
    for key, values, validity in _iter_training_arrays(frame, limb_mode):
        if values.ndim != 2:
            raise ValueError(f"Expected [N,C] while estimating stats for {key}, got {values.shape}")
        state = totals.setdefault(
            key,
            {
                "count": np.zeros(values.shape[1], dtype=np.float64),
                "sum": np.zeros(values.shape[1], dtype=np.float64),
                "sum_sq": np.zeros(values.shape[1], dtype=np.float64),
            },
        )
        for start in range(0, len(values), chunk_size):
            chunk = np.asarray(values[start : start + chunk_size], dtype=np.float64)
            finite = np.isfinite(chunk)
            if validity is not None:
                validity_chunk = np.asarray(validity[start : start + chunk_size], dtype=bool)
                if validity_chunk.shape != chunk.shape:
                    raise ValueError(
                        f"Validity shape {validity.shape} does not match sensor shape {values.shape}"
                    )
                finite &= validity_chunk
            safe = np.where(finite, chunk, 0.0)
            state["count"] += finite.sum(axis=0)
            state["sum"] += safe.sum(axis=0)
            state["sum_sq"] += (safe * safe).sum(axis=0)
    if mode == "train_rms_shared":
        total_count = sum(float(state["count"].sum()) for state in totals.values())
        total_sum_sq = sum(float(state["sum_sq"].sum()) for state in totals.values())
        if total_count <= 0:
            raise ValueError("No finite training samples were available for RMS normalization")
        return {"shared": {"rms": [float(np.sqrt(total_sum_sq / total_count))]}}
    if mode != "train_global":
        raise ValueError(f"Unsupported learned normalization mode: {mode}")
    groups = (
        (
            ("left_wrist", "right_wrist"),
            ("left_ankle", "right_ankle"),
            *(((("anchor",),) if "anchor" in totals else ())),
        )
        if limb_mode == "four"
        else (("left", "right"),)
    )
    pooled: dict[str, dict[str, np.ndarray]] = {}
    for keys in groups:
        available = [totals[key] for key in keys if key in totals]
        if not available:
            continue
        combined = {
            field: np.sum([state[field] for state in available], axis=0)
            for field in ("count", "sum", "sum_sq")
        }
        for key in keys:
            pooled[key] = combined
    output: dict[str, dict[str, list[float]]] = {}
    for key, state in pooled.items():
        if np.any(state["count"] <= 0):
            raise ValueError(f"No finite training samples for at least one {key} channel")
        count = np.maximum(state["count"], 1.0)
        mean = state["sum"] / count
        variance = np.maximum(state["sum_sq"] / count - mean * mean, 1e-12)
        output[key] = {"mean": mean.tolist(), "std": np.sqrt(variance).tolist()}
    if not output:
        raise ValueError("No training arrays were available for normalization")
    return output


class WindowManifestDataset(Dataset):
    """Load aligned bilateral windows stored as compressed NPZ files.

    Each NPZ must contain ``left`` and ``right`` arrays with shape [time, channels].
    An optional ``audio`` array may contain a log-mel spectrogram [mel, frames].
    Labels and participant IDs are intentionally kept in the manifest for fast splits.
    """

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        indices: Iterable[int] | None = None,
        normalization: str = "window_zscore",
        use_audio: bool = False,
        swap_probability: float = 0.0,
        limb_mode: str = "bilateral",
        normalization_stats: dict[str, dict[str, list[float]]] | None = None,
        normalization_scale: float = 8.0,
        missing_strategy: str = "error",
    ) -> None:
        self.frame = load_manifest(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        if indices is not None:
            self.frame = self.frame.iloc[list(indices)].reset_index(drop=True)
        self.normalization = normalization
        self.use_audio = use_audio
        self.limb_mode = limb_mode
        self.normalization_stats = normalization_stats
        self.normalization_scale = float(normalization_scale)
        self.missing_strategy = missing_strategy
        self.swap_probability = float(swap_probability)
        self._array_cache: dict[str, np.ndarray] = {}
        if not 0.0 <= self.swap_probability <= 1.0:
            raise ValueError("swap_probability must lie in [0, 1]")
        if self.normalization == "scale_only" and self.normalization_scale <= 0:
            raise ValueError("normalization_scale must be positive")

    def _normalize(self, array: np.ndarray, key: str) -> np.ndarray:
        if self.normalization == "window_zscore":
            return _window_zscore(array)
        if self.normalization == "scale_only":
            return array / self.normalization_scale
        if self.normalization == "train_global":
            if not self.normalization_stats or key not in self.normalization_stats:
                raise ValueError(f"Missing train-global normalization statistics for {key}")
            stats = self.normalization_stats[key]
            mean = np.asarray(stats["mean"], dtype=np.float32)[None, :]
            std = np.asarray(stats["std"], dtype=np.float32)[None, :]
            return (array - mean) / np.maximum(std, 1e-6)
        if self.normalization == "train_rms_shared":
            if not self.normalization_stats or "shared" not in self.normalization_stats:
                raise ValueError("Missing shared RMS normalization statistics")
            rms = float(self.normalization_stats["shared"]["rms"][0])
            return array / max(rms, 1e-6)
        if self.normalization in ("none", None):
            return array
        raise ValueError(f"Unknown normalization: {self.normalization}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_array_cache"] = {}
        return state

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[index]
        if self.limb_mode == "four":
            return self._get_four_limb(row)
        if self.limb_mode != "bilateral":
            raise ValueError(f"Unknown limb_mode: {self.limb_mode}")
        if set(BILATERAL_PATH_COLUMNS).issubset(self.frame.columns):
            return self._get_continuous_bilateral(row)
        path = Path(row.path)
        if not path.exists():
            raise FileNotFoundError(f"Window file not found: {path}")
        with np.load(path, allow_pickle=False) as sample:
            left = np.asarray(sample["left"], dtype=np.float32)
            right = np.asarray(sample["right"], dtype=np.float32)
            if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
                raise ValueError(
                    f"left/right must have equal [T,C] shapes, got {left.shape}/{right.shape}: {path}"
                )
            arrays, missing = repair_nonfinite_arrays(
                {"left": left, "right": right}, self.missing_strategy
            )
            left = self._normalize(arrays["left"], "left")
            right = self._normalize(arrays["right"], "right")

            if self.swap_probability and np.random.random() < self.swap_probability:
                left, right = right, left

            item: dict[str, torch.Tensor | str] = {
                "left": torch.from_numpy(left.copy()),
                "right": torch.from_numpy(right.copy()),
                "label": torch.tensor(int(row.label), dtype=torch.long),
                "participant": str(row.participant),
                "path": str(path),
                "missing_fraction": torch.tensor(
                    [missing["left"], missing["right"]], dtype=torch.float32
                ),
            }
            if self.use_audio:
                if "audio" not in sample:
                    raise KeyError(f"use_audio=true but NPZ has no 'audio': {path}")
                audio = np.asarray(sample["audio"], dtype=np.float32)
                if audio.ndim != 2:
                    raise ValueError(f"audio must have [mel,frames] shape: {path}")
                audio = (audio - audio.mean()) / max(float(audio.std()), 1e-6)
                item["audio"] = torch.from_numpy(audio.copy())
        return item

    def _get_continuous_bilateral(
        self, row: pd.Series
    ) -> dict[str, torch.Tensor | str]:
        start, stop = int(row.start), int(row.stop)
        arrays = {
            key: self._slice_with_edge_padding(
                self._cached_array(str(row[f"{key}_path"])), start, stop
            )
            for key in ("left", "right")
        }
        validity = {
            key: self._slice_with_edge_padding(
                self._cached_array(str(row[f"{key}_validity_path"])), start, stop
            ).astype(bool)
            for key in ("left", "right")
            if f"{key}_validity_path" in self.frame.columns
        }
        arrays, inferred_missing = repair_nonfinite_arrays(arrays, self.missing_strategy)
        missing = {
            key: float(1.0 - validity[key].mean())
            if key in validity
            else inferred_missing[key]
            for key in arrays
        }
        arrays = {key: self._normalize(value, key) for key, value in arrays.items()}
        if self.swap_probability and np.random.random() < self.swap_probability:
            arrays["left"], arrays["right"] = arrays["right"], arrays["left"]
            missing["left"], missing["right"] = missing["right"], missing["left"]
        item: dict[str, torch.Tensor | str] = {
            "left": torch.from_numpy(arrays["left"].copy()),
            "right": torch.from_numpy(arrays["right"].copy()),
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "participant": str(row.participant),
            "path": f"{row.left_path}:{start}:{stop}",
            "missing_fraction": torch.tensor(
                [missing["left"], missing["right"]], dtype=torch.float32
            ),
        }
        if "session" in row.index:
            item["session"] = str(row.session)
        if "center" in row.index:
            item["center"] = torch.tensor(int(row.center), dtype=torch.long)
        return item

    def _cached_array(self, path: str) -> np.ndarray:
        if path not in self._array_cache:
            self._array_cache[path] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._array_cache[path]

    @staticmethod
    def _slice_with_edge_padding(array: np.ndarray, start: int, stop: int) -> np.ndarray:
        target_length = stop - start
        source_start = max(start, 0)
        source_stop = min(stop, len(array))
        window = np.asarray(array[source_start:source_stop], dtype=np.float32)
        left_pad = max(0, -start)
        right_pad = max(0, stop - len(array))
        if left_pad or right_pad:
            if len(window) == 0:
                raise ValueError("Cannot pad a window with no source samples")
            window = np.pad(window, ((left_pad, right_pad), (0, 0)), mode="edge")
        if len(window) != target_length:
            raise ValueError(f"Window length mismatch: expected {target_length}, got {len(window)}")
        return window

    def _get_four_limb(self, row: pd.Series) -> dict[str, torch.Tensor | str]:
        if set(FOUR_LIMB_PATH_COLUMNS).issubset(self.frame.columns):
            start, stop = int(row.start), int(row.stop)
            path_columns = list(FOUR_LIMB_PATH_COLUMNS)
            if ANCHOR_PATH_COLUMN in self.frame.columns:
                path_columns.append(ANCHOR_PATH_COLUMN)
            arrays = {
                name.removesuffix("_path"): self._slice_with_edge_padding(
                    self._cached_array(str(row[name])), start, stop
                )
                for name in path_columns
            }
            validity_columns = list(FOUR_LIMB_VALIDITY_COLUMNS)
            if ANCHOR_VALIDITY_COLUMN in self.frame.columns:
                validity_columns.append(ANCHOR_VALIDITY_COLUMN)
            validity = {
                column.removesuffix("_validity_path"): self._slice_with_edge_padding(
                    self._cached_array(str(row[column])), start, stop
                ).astype(bool)
                for column in validity_columns
                if column in self.frame.columns
            }
            sample_path = f"{row.left_wrist_path}:{start}:{stop}"
        else:
            path = Path(row.path)
            if not path.exists():
                raise FileNotFoundError(f"Window file not found: {path}")
            with np.load(path, allow_pickle=False) as sample:
                required = {"left_wrist", "right_wrist", "left_ankle", "right_ankle"}
                if missing := required.difference(sample.files):
                    raise KeyError(f"Four-limb NPZ lacks {sorted(missing)}: {path}")
                arrays = {name: np.asarray(sample[name], dtype=np.float32) for name in required}
                if "anchor" in sample.files:
                    arrays["anchor"] = np.asarray(sample["anchor"], dtype=np.float32)
            sample_path = str(path)
            validity = {}
        shapes = {name: value.shape for name, value in arrays.items()}
        if len(set(shapes.values())) != 1 or len(next(iter(shapes.values()))) != 2:
            raise ValueError(f"Four limbs must have equal [T,C] shapes, got {shapes}")
        arrays, inferred_missing = repair_nonfinite_arrays(arrays, self.missing_strategy)
        missing = {
            key: float(1.0 - validity[key].mean()) if key in validity else inferred_missing[key]
            for key in arrays
        }
        arrays = {name: self._normalize(value, name) for name, value in arrays.items()}
        if self.swap_probability:
            if np.random.random() < self.swap_probability:
                arrays["left_wrist"], arrays["right_wrist"] = (
                    arrays["right_wrist"],
                    arrays["left_wrist"],
                )
            if np.random.random() < self.swap_probability:
                arrays["left_ankle"], arrays["right_ankle"] = (
                    arrays["right_ankle"],
                    arrays["left_ankle"],
                )
        item: dict[str, torch.Tensor | str] = {
            name: torch.from_numpy(value.copy()) for name, value in arrays.items()
        }
        item.update(
            label=torch.tensor(int(row.label), dtype=torch.long),
            participant=str(row.participant),
            path=sample_path,
            missing_fraction=torch.tensor(
                [
                    missing[key]
                    for key in (
                        "left_wrist",
                        "right_wrist",
                        "left_ankle",
                        "right_ankle",
                        *(("anchor",) if "anchor" in missing else ()),
                    )
                ],
                dtype=torch.float32,
            ),
        )
        if "session" in row.index:
            item["session"] = str(row.session)
        if "center" in row.index:
            item["center"] = torch.tensor(int(row.center), dtype=torch.long)
        return item


@dataclass(frozen=True)
class ParticipantSplit:
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    test_participant: str
    val_participant: str


def build_participant_split(
    frame: pd.DataFrame,
    test_participant: str,
    val_participant: str | None = None,
) -> ParticipantSplit:
    """Create a leakage-free participant split for one LOPO fold."""
    participants = sorted(frame["participant"].astype(str).unique().tolist())
    test_participant = str(test_participant)
    if test_participant not in participants:
        raise ValueError(f"Unknown test participant {test_participant}; choices={participants}")
    candidates = [p for p in participants if p != test_participant]
    if not candidates:
        raise ValueError("At least three participants are recommended for train/val/test splitting")
    if val_participant is None:
        test_position = participants.index(test_participant)
        val_participant = candidates[test_position % len(candidates)]
    val_participant = str(val_participant)
    if val_participant not in candidates:
        raise ValueError("Validation participant must exist and differ from the test participant")
    p = frame["participant"].astype(str)
    test = frame.index[p == test_participant].tolist()
    val = frame.index[p == val_participant].tolist()
    train = frame.index[(p != test_participant) & (p != val_participant)].tolist()
    if not train or not val or not test:
        raise ValueError("Participant split produced an empty partition")
    return ParticipantSplit(train, val, test, test_participant, val_participant)
