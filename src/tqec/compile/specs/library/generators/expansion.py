"""Defines the layers for implementing the expansion of a surface code patch."""

from collections.abc import Mapping
from dataclasses import replace

import numpy

from tqec.circuit.measurement import Measurement
from tqec.circuit.qubit import GridQubit
from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
from tqec.compile.specs.library.generators import FixedBulkConventionGenerator
from tqec.compile.specs.library.generators.fixed_boundary import FixedBoundaryConventionGenerator
from tqec.compile.specs.library.generators.utils import PlaquetteMapper
from tqec.plaquette import RPNGDescription
from tqec.plaquette.plaquette import Plaquettes
from tqec.plaquette.rpng import ExtendedBasis, PauliBasis
from tqec.templates.expansion import ExpansionTemplate
from tqec.utils.enums import Orientation
from tqec.utils.frozendefaultdict import FrozenDefaultDict


class SurfaceCodeExpansionLayer(PlaquetteLayer):
    def __init__(
        self,
        generator: FixedBulkConventionGenerator | FixedBoundaryConventionGenerator,
        mapper: PlaquetteMapper,
        orientation: Orientation,
        mirrored_horizontal: bool,
        source_distance: int,
    ):
        """Construct a plaquette layer representing the surface code expansion.

        Args:
            generator: ConventionGenerator to be used
            mapper: PlaquetteMapper to be used
            orientation: the orientation of the surface code patches
            mirrored_horizontal: whether HORIZONTAL surface code patches are mirrored (x -> -x),
                placing the cultivation in the bottom-left corner instead of the bottom-right.
                Ignored for VERTICAL patches, which are transposed instead.
            source_distance: the code distance of the source surface code patch

        """
        template = ExpansionTemplate(orientation, mirrored_horizontal, source_distance)

        # Compute the RPNG descriptions for the plaquettes of the expansion layer.
        descriptions: dict[int, RPNGDescription] = {}
        if isinstance(generator, FixedBulkConventionGenerator):
            memory = generator.get_memory_qubit_rpng_descriptions(orientation)
        else:
            memory = generator.get_memory_qubit_rpng_descriptions(False, orientation)

        plaquette_resets = template.get_plaquette_resets()

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

        super().__init__(template, Plaquettes(rpng_descriptions.map_values(mapper.get_plaquette)))

        # The basis of the stabilizer measured by each plaquette of the template.
        self._bases: dict[int, PauliBasis | None] = {
            label: next((corner.p for corner in description.corners if corner.p), None)
            for label, description in descriptions.items()
        }

    def observable_correction(
        self, representative: Mapping[GridQubit, PauliBasis], k: int
    ) -> frozenset[Measurement]:
        """Return the measurements connecting a representative to the logical lines of the patch.

        The representative of the source patch is L-shaped: an arm of X and an arm of Z extending
        from a Y on its corner. The logical lines of the target patch run through its middle
        instead. Each arm maps onto the middle line parallel to it through the stabilizers of
        its basis lying strictly between both lines, which are all measured by this layer.

        Args:
            representative: the Pauli basis of each qubit of the representative, in the
                coordinates of the block.
            k: scaling factor.

        Returns:
            the measurements, with qubits in the coordinates of the block, and offsets relative
            to the end of this layer.

        """
        middle = 2 * k + 1
        corner = next(qubit for qubit, basis in representative.items() if basis == PauliBasis.Y)
        labels = self.template.instantiate(k)
        increments = self.template.get_increments()

        measurements: set[Measurement] = set()
        for arm in (PauliBasis.X, PauliBasis.Z):
            qubit = next(q for q, basis in representative.items() if basis == arm)
            along_column = qubit.x == corner.x
            line = corner.x if along_column else corner.y
            low, high = min(line, middle), max(line, middle)
            for row, col in numpy.ndindex(labels.shape):
                syndrome = GridQubit(col * increments.x, row * increments.y)
                position = syndrome.x if along_column else syndrome.y
                if low < position < high and self._bases.get(int(labels[row, col])) == arm:
                    measurements.add(Measurement(syndrome, -1))

        return frozenset(measurements)
