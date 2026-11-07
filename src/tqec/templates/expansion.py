"""Defines templates representing the expansion of logical qubits."""

from collections.abc import Sequence

import numpy
from numpy import typing as npt
from typing_extensions import override

from tqec.templates.base import BorderIndices
from tqec.templates.enums import TemplateBorder
from tqec.templates.qubit import QubitTemplate
from tqec.utils import Orientation
from tqec.utils.scale import LinearFunction, PlaquetteScalable2D


class ExpansionTemplate(QubitTemplate):
    def __init__(self, orientation: Orientation, source_distance: int):
        """Initialize an expansion layer template for growing a surface code patch.

        Args:
            orientation: the orientation of the source surface code patch
            source_distance: the code distance of the source surface code patch

        The source surface code patch will be grown into a target surface code patch by activating
        additional plaquettes around its left and upper boundaries.

        """
        super().__init__()

        self._orientation = orientation
        self._source_distance = source_distance

    def _inside_source(self, target_distance: int, row: int, col: int) -> int:
        """Convert the row/col to describe its location w.r.t. the source patch.

        Args:
            target_distance: the code distance of the target patch
            row: the row of the plaquette
            col: the col of the plaquette

        Returns:
            0 if the plaquette has all its data qubits outside the source patch
            1 if the plaquette has some data qubits outside the source patch
            2 if the plaquette has all its data qubits inside the source patch

        """
        extra_count = target_distance - self._source_distance + 1

        if row >= extra_count and col >= extra_count:
            return 2
        elif row >= extra_count - 1 and col >= extra_count - 1:
            return 1
        else:
            return 0

    @staticmethod
    def _along_boundary(target_distance: int, row: int, col: int) -> int:
        """Convert the row/col to describe its location w.r.t. boundaries of the target patch.

        Args:
            target_distance: the code distance of the target patch
            row: the row of the plaquette
            col: the col of the plaquette

        Returns:
            0 if the plaquette does not lie along any boundaries of the target patch
            1 if the plaquette lies along a horizontal boundary of the target patch
            2 if the plaquette lies along a vertical boundary of the target patch
            3 if the plaquette lies along both boundaries of the target patch (i.e. corner)

        """
        return int(row in (0, target_distance)) | int(col in (0, target_distance)) << 1

    @staticmethod
    def _against_diagonal(row: int, col: int) -> int:
        """Convert the row/col to describe its location w.r.t. the diagonal of the target patch.

        Args:
            row: the row of the plaquette
            col: the col of the plaquette

        Returns:
            0 if the plaquette lies strictly above the diagonal of the target patch

        """
        diagonal = 0

        if row in (col, col + 1):
            diagonal += 1
        elif row < col:
            diagonal += 2

        return diagonal

    @override
    def instantiate(
        self, k: int, plaquette_indices: Sequence[int] | None = None
    ) -> npt.NDArray[numpy.int_]:
        if plaquette_indices is None:
            plaquette_indices = [i + 1 for i in range(self.expected_plaquettes_number)]

        layout = numpy.zeros(self.shape(k).to_numpy_shape(), dtype=numpy.int_)
        target_distance = 2 * k + 1

        for row, col in numpy.ndindex(layout.shape):
            color = (row + col) % 2
            coordinates = (col, row) if self._orientation == Orientation.VERTICAL else (row, col)

            source = self._inside_source(target_distance, *coordinates)
            boundary = self._along_boundary(target_distance, *coordinates)
            diagonal = (source != 2) * self._against_diagonal(*coordinates)

            index = (diagonal << 5) | (boundary << 3) | (source << 1) | color

            layout[row, col] = plaquette_indices[index]

        return layout

    @override
    @property
    def scalable_shape(self) -> PlaquetteScalable2D:
        return PlaquetteScalable2D(LinearFunction(2, 2), LinearFunction(2, 2))

    @property
    @override
    def expected_plaquettes_number(self) -> int:
        """The number of different types of plaquettes that the target patch can have.

        Note that only 32 of those values appear in the instantiated plaquettes.
        """
        return 128

    @override
    def get_border_indices(self, border: TemplateBorder) -> BorderIndices:
        raise NotImplementedError("Surface code expansion is not meant to be trimmed by pipes.")
