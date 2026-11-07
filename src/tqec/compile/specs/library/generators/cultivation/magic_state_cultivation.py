import itertools
from pathlib import Path
from typing import cast

import numpy
import stim

from tqec.circuit.moment import Moment
from tqec.circuit.qubit_map import QubitMap
from tqec.circuit.schedule import ScheduledCircuit
from tqec.compile.blocks.block import Block
from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
from tqec.compile.blocks.layers.composed import RepeatedLayer, SequencedLayers
from tqec.compile.specs.base import CubeSpec
from tqec.compile.specs.library.generators import (
    FixedBoundaryConventionGenerator,
    FixedBulkConventionGenerator,
)
from tqec.compile.specs.library.generators.cultivation.surface_code_expansion import (
    SurfaceCodeExpansionLayer,
)
from tqec.compile.specs.library.generators.utils import PlaquetteMapper
from tqec.computation.cube import CultivationBlock
from tqec.plaquette import (
    FrozenDefaultDict,
    Plaquette,
    PlaquetteQubits,
    Plaquettes,
    RPNGDescription,
)
from tqec.plaquette.debug import DrawPolygon, PlaquetteDebugInformation
from tqec.plaquette.rpng import PauliBasis
from tqec.templates.qubit import QubitTemplate
from tqec.utils import Orientation
from tqec.utils.scale import LinearFunction


def _batched(iterable, n):
    it = iter(iterable)
    while chunk := tuple(itertools.islice(it, n)):
        yield chunk


class CultivationGenerator:
    __STATIC_CIRCUIT = Path(__file__).parent / "magic-state-cultivation-s-state.stim"
    __STEANE_ANCHOR = (3.0, 1.0)
    __STEANE_CORNER = (5.5, 7.5)

    __SLICE_SIZE = 7
    __BLOCK_ONE_SLICES = 7

    def __init__(
        self,
        generator: FixedBulkConventionGenerator | FixedBoundaryConventionGenerator,
        mapper: PlaquetteMapper,
    ) -> None:
        """Contain the circuitry generation for an attempt of Magic State Cultivation.

        Args:
            generator: Generator used to produce the memory rounds at the end of the cultivation
            mapper: PlaquetteMapper to be used

        """
        self._generator = generator
        self._mapper = mapper
        self._expander = SurfaceCodeExpansionLayer(generator, mapper)

    @staticmethod
    def __load_static_cultivation_circuit(orientation: Orientation) -> ScheduledCircuit:
        """Prepare the static circuit (Injection+Superdense+Doublecheck+Teleportation)."""
        anchor = numpy.array(CultivationGenerator.__STEANE_ANCHOR)
        corner = numpy.array(CultivationGenerator.__STEANE_CORNER)

        loaded = stim.Circuit.from_file(CultivationGenerator.__STATIC_CIRCUIT)
        static = stim.Circuit()

        for instruction in loaded.flattened():
            assert isinstance(instruction, stim.CircuitInstruction)
            if instruction.name == "QUBIT_COORDS":
                location = numpy.array(instruction.gate_args_copy())
                position = 2 * (location - (anchor + corner))
                if orientation == Orientation.VERTICAL:
                    position[0], position[1] = position[1], position[0]
                static.append("QUBIT_COORDS", instruction.targets_copy(), position)
            elif instruction.name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                static.append(instruction)

        return ScheduledCircuit.from_circuit(static)

    @staticmethod
    def __slice_into_subcircuits(circuit: ScheduledCircuit) -> list[ScheduledCircuit]:
        """Slice a circuit into pieces of the required size."""
        return [
            ScheduledCircuit(
                [m for _, m in subcircuit],
                list(range(CultivationGenerator.__SLICE_SIZE)),
                circuit.qubit_map,
            )
            for subcircuit in _batched(circuit.scheduled_moments, CultivationGenerator.__SLICE_SIZE)
        ]

    def __empty_plaquette(self) -> Plaquette:
        return self._mapper.get_plaquette(RPNGDescription.empty())

    def __idling_layer(self, template: QubitTemplate) -> PlaquetteLayer:
        idle = Plaquette(
            "IDLING_LAYER",
            PlaquetteQubits([], []),
            ScheduledCircuit(
                [Moment(stim.Circuit("SHIFT_COORDS"))],
                [CultivationGenerator.__SLICE_SIZE - 1],
                QubitMap(),
            ),
        )
        return PlaquetteLayer(
            template,
            Plaquettes(FrozenDefaultDict({4: idle}, default_value=self.__empty_plaquette())),
        )

    def __stabilizing_layer(
        self, template: QubitTemplate, orientation: Orientation
    ) -> PlaquetteLayer:
        if isinstance(self._generator, FixedBulkConventionGenerator):
            plaquettes = self._generator.get_memory_qubit_plaquettes(orientation)
        else:  # if isinstance(self._generator, FixedBoundaryConventionGenerator):
            plaquettes = self._generator.get_memory_qubit_plaquettes(False, orientation)

        return PlaquetteLayer(template, plaquettes)

    def get_cultivation_blocks(self, spec: CubeSpec, height: LinearFunction) -> list[Block]:
        """Obtain the sequence of Block implementing a Magic State Cultivation.
        n.b. The cultivation process requires at least 3 Block and a target distance of at least 7.

        Returns:
            a sequence of Block representing a single attempt of the cultivation process.

        """
        assert isinstance(spec.kind, CultivationBlock)

        orientation = spec.kind.get_orientation()

        static_circuit = CultivationGenerator.__load_static_cultivation_circuit(orientation)
        static_subcircuits = CultivationGenerator.__slice_into_subcircuits(static_circuit)

        # fmt: off
        patch = DrawPolygon(
            {
                PauliBasis.Y: [
                    static_circuit.qubit_map.i2q[i]
                    for i in [
                        14, 22, 38, 39, 46, 47, 57, 63, 68, 64, 65, 69, 66, 67, 62,
                        56, 45, 37, 29, 13,  5, 11,  4, 10,  9,  3,  0,  2,  1,  6,
                    ]
                ]
            }
        )
        # fmt: on

        template = cast(QubitTemplate, self._generator.get_memory_qubit_raw_template())

        static_layers_one = []
        static_layers_two = []
        for index, subcircuit in enumerate(static_subcircuits):
            plaquette = Plaquette(
                f"MSC_{index}",
                PlaquetteQubits([], sorted(subcircuit.qubits)),
                subcircuit,
                debug_information=PlaquetteDebugInformation(draw_polygons=patch),
            )

            layer = PlaquetteLayer(
                template,
                Plaquettes(
                    FrozenDefaultDict({4: plaquette}, default_value=self.__empty_plaquette())
                ),
            )

            if index < CultivationGenerator.__BLOCK_ONE_SLICES:
                static_layers_one.append(layer)
            else:
                static_layers_two.append(layer)

        # Prepare Cultivation Block #1
        idling_layers = RepeatedLayer(
            self.__idling_layer(template), height - (len(static_layers_one) - 2)
        )
        block_one = Block(
            [
                static_layers_one[0],
                SequencedLayers([*static_layers_one[1:-1], idling_layers]),
                static_layers_one[-1],
            ]
        )

        # Prepare Cultivation Block #2
        expansion = self._expander.get_plaquette_layer(orientation, 5)
        stabilizer = self.__stabilizing_layer(template, orientation)
        stabilizing_layers = RepeatedLayer(stabilizer, height - len(static_layers_two))
        block_two = Block(
            [
                static_layers_two[0],
                SequencedLayers([*static_layers_two[1:], expansion, stabilizing_layers]),
                stabilizer,
            ]
        )

        return [block_one, block_two]
