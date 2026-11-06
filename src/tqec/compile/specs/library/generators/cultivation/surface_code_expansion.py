"""Defines the layers for implementing the expansion of a surface code patch."""

from dataclasses import replace

import numpy
from numpy import typing as npt

from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
from tqec.compile.specs.library.generators import FixedBulkConventionGenerator
from tqec.compile.specs.library.generators.fixed_boundary import (
    FixedBoundaryConventionGenerator,
)
from tqec.compile.specs.library.generators.utils import PlaquetteMapper
from tqec.plaquette import RPNGDescription
from tqec.plaquette.plaquette import Plaquettes
from tqec.plaquette.rpng import ExtendedBasis
from tqec.templates.expansion import ExpansionDirection, ExpansionTemplate
from tqec.templates.qubit import QubitTemplate
from tqec.utils import TQECError
from tqec.utils.enums import Basis, Orientation
from tqec.utils.frozendefaultdict import FrozenDefaultDict

ResetKey = tuple[Basis | None, Basis | None, Basis | None, Basis | None]


class SurfaceCodeExpansionLayer(PlaquetteLayer):
    def __init__(
        self,
        generator: FixedBulkConventionGenerator | FixedBoundaryConventionGenerator,
        mapper: PlaquetteMapper,
    ):
        """Create an expansion layer for growing a surface code."""
        self._generator = generator
        self._mapper = mapper

    @staticmethod
    def _get_data_resets(
        orientation: Orientation,
        target_distance: int,
        source_distance: int,
        direction: ExpansionDirection,
    ) -> npt.NDArray[numpy.object_]:
        """Compute the RESETs that occurs at every data qubits of the target code patch."""
        upper = Basis.X if orientation == Orientation.HORIZONTAL else Basis.Z
        rows, cols = numpy.indices((target_distance, target_distance))

        # Prepare the RESETs for the LU case
        resets = numpy.where(cols >= rows, upper, upper.flipped()).astype(object)
        resets[target_distance - source_distance :, target_distance - source_distance :] = None

        # Shift the RESETs around to bring the LU case to the RU, RD, LD expansion directions.
        if direction in (ExpansionDirection.RD, ExpansionDirection.LD):
            resets = resets[::-1, :]
        if direction in (ExpansionDirection.RU, ExpansionDirection.RD):
            resets = resets[:, ::-1]

        return resets

    @staticmethod
    def _get_plaquette_resets(
        template: ExpansionTemplate,
        orientation: Orientation,
        source_distance: int,
        direction: ExpansionDirection,
    ) -> dict[int, tuple[int, ResetKey]]:
        """Compute the RESETs that occurs for every plaquette of the target code patch."""
        resets: dict[int, tuple[int, ResetKey]] = {}

        # The minimal value to get all plaquette indices that are used in the ExpansionTemplate.
        k: int = 4
        target_distance: int = 2 * k + 1
        expansion_indices = template.instantiate(k)
        plaquette_indices = QubitTemplate.instantiate(template, k)
        data_resets = SurfaceCodeExpansionLayer._get_data_resets(
            orientation, target_distance, source_distance, direction
        )
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

    def get_plaquette_layer(
        self, orientation: Orientation, source_distance: int, direction: ExpansionDirection
    ) -> PlaquetteLayer:
        """Construct a plaquette layer representing the surface code expansion."""
        template = ExpansionTemplate(orientation, source_distance, direction)

        # Compute the RPNG descriptions for the plaquettes of the expansion layer.
        descriptions: dict[int, RPNGDescription] = {}
        if isinstance(self._generator, FixedBulkConventionGenerator):
            memory = self._generator.get_memory_qubit_rpng_descriptions(orientation)
        else:
            memory = self._generator.get_memory_qubit_rpng_descriptions(False, orientation)

        plaquette_resets = self._get_plaquette_resets(
            template, orientation, source_distance, direction
        )

        for label, (slot, resets) in plaquette_resets.items():
            description = memory[slot]
            corners = tuple(
                corner
                if basis is None or corner.p is None
                else replace(corner, r=ExtendedBasis(basis.value.lower()))
                for corner, basis in zip(description.corners, resets)
            )
            descriptions[label] = replace(description, corners=corners)
        rpng_descriptions = FrozenDefaultDict(descriptions, default_value=RPNGDescription.empty())

        return PlaquetteLayer(
            template, Plaquettes(rpng_descriptions.map_values(self._mapper.get_plaquette))
        )
