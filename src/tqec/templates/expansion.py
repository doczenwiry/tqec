"""Defines templates representing the expansion of logical qubits."""

from collections.abc import Sequence

import numpy
from numpy import typing as npt
from typing_extensions import override

from tqec.templates.base import BorderIndices
from tqec.templates.enums import TemplateBorder
from tqec.templates.qubit import QubitTemplate
from tqec.utils import Orientation, TQECError
from tqec.utils.enums import Basis
from tqec.utils.scale import LinearFunction, PlaquetteScalable2D

ResetKey = tuple[Basis | None, Basis | None, Basis | None, Basis | None]


class ExpansionTemplate(QubitTemplate):
    def __init__(self, orientation: Orientation, mirrored_horizontal: bool, source_distance: int):
        """Initialize an expansion layer template for growing a surface code patch.

        Args:
            orientation: the orientation of the source surface code patch
            mirrored_horizontal: whether HORIZONTAL surface code patches are mirrored (x -> -x),
                placing the cultivation in the bottom-left corner instead of the bottom-right.
                Ignored for VERTICAL patches, which are transposed instead.
            source_distance: the code distance of the source surface code patch

        The source surface code patch will be grown into a target surface code patch by activating
        additional plaquettes around its left and upper boundaries.

        """
        super().__init__()

        self._orientation = orientation
        # Mirroring only applies to HORIZONTAL patches; VERTICAL ones are transposed instead.
        self._mirrored = mirrored_horizontal and orientation == Orientation.HORIZONTAL
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
            1 if the plaquette lies along only the horizontal boundary of the target patch
            2 if the plaquette lies along only the vertical boundary of the target patch
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
            1 if the plaquette lies on or right below the diagonal of the target patch
            2 if the plaquette lies strictly below those diagonals of the target patch

        """
        if row < col:
            return 2
        elif col <= row <= col + 1:
            return 1
        else:
            return 0

    @override
    def instantiate(
        self, k: int, plaquette_indices: Sequence[int] | None = None
    ) -> npt.NDArray[numpy.int_]:
        target_distance = 2 * k + 1

        if target_distance <= self._source_distance:
            raise TQECError(
                f"Expanding a distance-{self._source_distance} patch requires a strictly "
                f"larger target distance, got {target_distance} (k={k})."
            )

        if plaquette_indices is None:
            plaquette_indices = [i + 1 for i in range(self.expected_plaquettes_number)]

        layout = numpy.zeros(self.shape(k).to_numpy_shape(), dtype=numpy.int_)

        for row, col in numpy.ndindex(layout.shape):
            color = (row + col) % 2

            if self._orientation == Orientation.VERTICAL:
                coordinates = (col, row)
            elif self._mirrored:
                coordinates = (row, target_distance - col)
            else:
                coordinates = (row, col)

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

    def _get_data_resets(self, target_distance: int) -> npt.NDArray[numpy.object_]:
        """Compute the RESETs that occurs at every data qubits of the target code patch."""
        rows, cols = numpy.indices((target_distance, target_distance))

        # Prepare the RESETs for the LU case
        resets = numpy.where(cols >= rows, Basis.X, Basis.Z).astype(object)
        resets[
            target_distance - self._source_distance :, target_distance - self._source_distance :
        ] = None

        if self._orientation == Orientation.VERTICAL:
            return resets.T
        else:
            return resets[:, ::-1] if self._mirrored else resets

    def get_plaquette_resets(self) -> dict[int, tuple[int, ResetKey]]:
        """Compute the RESETs that occurs for every plaquette of the target code patch."""
        resets: dict[int, tuple[int, ResetKey]] = {}

        # The minimal value to get all plaquette indices that are used in the ExpansionTemplate.
        k: int = 4
        target_distance: int = 2 * k + 1
        expansion_indices = self.instantiate(k)
        plaquette_indices = QubitTemplate.instantiate(self, k)
        data_resets = self._get_data_resets(target_distance)
        padded = numpy.pad(data_resets, 1, constant_values=None)
        for row, col in numpy.ndindex(expansion_indices.shape):
            corners = tuple(padded[row : row + 2, col : col + 2].ravel())
            kind = (int(plaquette_indices[row, col]), corners)
            known = resets.setdefault(int(expansion_indices[row, col]), kind)
            if known != kind:
                raise TQECError(
                    f"Label {expansion_indices[row, col]} stands for both {known} and {kind}."
                )

        return resets
