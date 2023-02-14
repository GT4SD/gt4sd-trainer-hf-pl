#!/usr/bin/env python
#
# MIT License
#
# Copyright (c) 2023 GT4SD team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""Run training pipelines for the GT4SD."""

import logging
import sys
from dataclasses import dataclass, field
from typing import IO, Iterable, Optional, Tuple, cast

from .argument_parser import ArgumentParser, DataClass, DataClassType
from .core import (
    LanguageModelingDataArguments,
    LanguageModelingModelArguments,
    LanguageModelingTrainingPipeline,
)
from .pytorch_lightning_trainer import PytorchLightningTrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class TrainerArguments:
    """Trainer arguments."""

    __name__ = "trainer_base_args"

    configuration_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Configuration file for the training in JSON format. It can be used to completely by-pass pipeline specific arguments."
        },
    )


class TrainerArgumentParser(ArgumentParser):
    """Argument parser using a custom help logic."""

    def print_help(self, file: Optional[IO[str]] = None) -> None:
        """Print help checking dynamically whether a specific pipeline is passed.
        Args:
            file: an optional I/O stream. Defaults to None, a.k.a., stdout and stderr.
        """
        try:
            help_args_set = {"-h", "--help"}
            if (
                len(set(sys.argv).union(help_args_set)) < len(help_args_set) + 2
            ):  # considering filename
                super().print_help()
                return
            args = [arg for arg in sys.argv if arg not in help_args_set]
            parsed_arguments = super().parse_args_into_dataclasses(
                args=args, return_remaining_strings=True
            )
            trainer_arguments = None
            for arguments in parsed_arguments:
                if arguments.__name__ == "trainer_base_args":
                    trainer_arguments = arguments
                    break
            if trainer_arguments:
                parser = ArgumentParser(
                    tuple(
                        [
                            TrainerArguments,
                            PytorchLightningTrainingArguments,
                            LanguageModelingDataArguments,
                            LanguageModelingModelArguments,
                        ]  # type:ignore
                    )
                )
                parser.print_help()
        except Exception:
            super().print_help()

    def parse_json_file(self, json_file: str) -> Tuple[DataClass, ...]:  # type: ignore
        """Overriding default .json parser.
        It by-passes all command line arguments and simply add the training pipeline.
        Args:
            json_file: JSON file containing pipeline configuration parameters.
        Returns:
            parsed arguments in a tuple of dataclasses.
        """
        number_of_dataclass_types = len(self.dataclass_types)  # type:ignore
        self.dataclass_types = [
            dataclass_type
            for dataclass_type in self.dataclass_types  # type:ignore
            if "gt4sd.trainer.hf_pl.TrainerArguments" not in str(dataclass_type)
        ]
        try:
            parsed_arguments = super().parse_json_file(  # type:ignore
                json_file=json_file, allow_extra_keys=True
            )
        except Exception:
            logger.exception(
                f"error parsing configuration file: {json_file}, printing error and exiting"
            )
            sys.exit(1)
        if number_of_dataclass_types > len(self.dataclass_types):
            self.dataclass_types.insert(0, cast(DataClassType, TrainerArguments))
        return parsed_arguments


def main() -> None:
    """
    Run a training pipeline.
    Raises:
        ValueError: in case the provided training pipeline provided is not supported.
    """

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    base_args = TrainerArgumentParser(
        cast(DataClassType, TrainerArguments)
    ).parse_args_into_dataclasses(return_remaining_strings=True)[0]

    parser = TrainerArgumentParser(
        cast(
            Iterable[DataClassType],
            tuple(
                [
                    TrainerArguments,
                    PytorchLightningTrainingArguments,
                    LanguageModelingDataArguments,
                    LanguageModelingModelArguments,
                ]
            ),
        )
    )

    configuration_filepath = base_args.configuration_file
    if configuration_filepath:
        args = parser.parse_json_file(json_file=configuration_filepath)
    else:
        args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    config = {
        arg.__name__: arg.__dict__
        for arg in args
        if not isinstance(arg, TrainerArguments)
        and not isinstance(arg, list)
        and isinstance(arg.__name__, str)
    }

    pipeline = LanguageModelingTrainingPipeline()
    pipeline.train(**config)


if __name__ == "__main__":
    main()
