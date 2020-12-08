import csv
import pathlib
from collections import defaultdict
from typing import Dict, Mapping, Sequence, Tuple, Union

import termcolor
import torch

# This logger is based on https://github.com/denisyarats/pytorch_sac/blob/master/logger.py
# with some modifications and some of its features removed


LogFormatType = Sequence[Tuple[str, str, str]]
LogTypes = Union[int, float, torch.Tensor]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0.0
        self._count = 0

    def update(self, value: float, n: int = 1):
        self._sum += value
        self._count += n

    def value(self) -> float:
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name: Union[str, pathlib.Path], formatting: LogFormatType):
        self._csv_file_path = self._prepare_file(file_name, ".csv")
        self._formatting = formatting
        self._meters: Dict[str, AverageMeter] = defaultdict(AverageMeter)
        self._csv_file = open(self._csv_file_path, "w")
        self._csv_writer = None

    @staticmethod
    def _prepare_file(prefix: Union[str, pathlib.Path], suffix: str) -> pathlib.Path:
        file_path = pathlib.Path(prefix).with_suffix(suffix)
        if file_path.exists():
            file_path.unlink()
        return file_path

    def log(self, key: str, value: float):
        self._meters[key].update(value)

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=sorted(data.keys()), restval=0.0
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    @staticmethod
    def _format(key: str, value: float, format_type: str):
        if format_type == "int":
            value = int(value)
            return f"{key}: {value}"
        elif format_type == "float":
            return f"{key}: {value:.04f}"
        elif format_type == "time":
            return f"{key}: {value:04.1f} s"
        else:
            raise ValueError(f"Invalid format type: {format_type}")

    def _dump_to_console(self, data, prefix: str, color: str = "yellow"):
        prefix = termcolor.colored(prefix, color)
        pieces = [f"| {prefix: <14}"]
        for key, disp_key, ty in self._formatting:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(" | ".join(pieces))

    def dump(self, step: int, prefix: str, save: bool = True, color: str = "yellow"):
        if len(self._meters) == 0:
            return
        if save:
            data = dict([(key, meter.value()) for key, meter in self._meters.items()])
            data["step"] = step
            self._dump_to_csv(data)
            self._dump_to_console(data, prefix, color)
        self._meters.clear()


class Logger(object):
    def __init__(
        self,
        log_dir: str,
    ):
        self._log_dir = pathlib.Path(log_dir)
        self._groups: Dict[str, Tuple[MetersGroup, int, str]] = {}
        self._step = 0

    def register_group(
        self,
        group_name: str,
        log_format: LogFormatType,
        dump_frequency: int = 1,
        color: str = "yellow",
    ):
        if group_name in self._groups:
            print(f"Group {group_name} has already been registered.")
            return
        new_group = MetersGroup(self._log_dir / group_name, formatting=log_format)
        self._groups[group_name] = (new_group, dump_frequency, color)

    def log(self, group_name: str, data: Mapping[str, LogTypes]):
        if group_name not in self._groups:
            raise ValueError(f"Group {group_name} has not been registered.")
        meter_group, dump_frequency, color = self._groups[group_name]
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                value = value.item()  # type: ignore
            meter_group.log(key, value)
        self._step += 1
        if self._step % dump_frequency == 0:
            self.dump(group_name)

    def dump(self, group_name: str, save: bool = True):
        if group_name not in self._groups:
            raise ValueError(f"Group {group_name} has not been registered.")
        meter_group, dump_frequency, color = self._groups[group_name]
        meter_group.dump(self._step, group_name, save, color=color)
