"""Defines the layers for implementing the expansion of a surface code patch."""

from dataclasses import replace

from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
from tqec.compile.specs.library.generators import FixedBulkConventionGenerator
from tqec.compile.specs.library.generators.fixed_boundary import FixedBoundaryConventionGenerator
from tqec.compile.specs.library.generators.utils import PlaquetteMapper
from tqec.plaquette import RPNGDescription
from tqec.plaquette.plaquette import Plaquettes
from tqec.plaquette.rpng import ExtendedBasis
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
