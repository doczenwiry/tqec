import itertools
from pathlib import Path
from typing import ClassVar

import numpy
import stim

from tqec.circuit.schedule import ScheduledCircuit
from tqec.compile.blocks.block import Block
from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
from tqec.compile.blocks.layers.composed import RepeatedLayer, SequencedLayers
from tqec.plaquette import (
    FrozenDefaultDict,
    Plaquette,
    PlaquetteQubits,
    Plaquettes,
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


class CultivationBlock:
    __STATIC_CIRCUIT: Path = Path(__file__).parent / "magic-state-cultivation-s-state.stim"
    __STEANE_ANCHOR: tuple[float, float] = (3.0, 1.0)
    __STEANE_CORNER: tuple[float, float] = (5.5, 7.5)

    __BLOCK_ONE_SLICES: int = 7
    __CULTIVATION_PATCH: ClassVar = [
        14,
        22,
        38,
        39,
        46,
        47,
        57,
        63,
        68,
        64,
        65,
        69,
        66,
        67,
        62,
        56,
        45,
        37,
        29,
        13,
        5,
        11,
        4,
        10,
        9,
        3,
        0,
        2,
        1,
        6,
    ]

    @staticmethod
    def __load_static_cultivation_circuit(orientation: Orientation) -> ScheduledCircuit:
        """Prepare the static circuit (Injection+Superdense+Doublecheck+Teleportation)."""
        anchor = numpy.array(CultivationBlock.__STEANE_ANCHOR)
        corner = numpy.array(CultivationBlock.__STEANE_CORNER)

        loaded = stim.Circuit.from_file(CultivationBlock.__STATIC_CIRCUIT)
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
    def __slice_into_subcircuits(circuit: ScheduledCircuit, size: int) -> list[ScheduledCircuit]:
        """Slice a circuit into pieces of the required size."""
        return [
            ScheduledCircuit(
                [m for _, m in subcircuit],
                list(range(size)),
                circuit.qubit_map,
            )
            for subcircuit in _batched(circuit.scheduled_moments, size)
        ]

    @staticmethod
    def get_cultivation_blocks(
        orientation: Orientation,
        height: LinearFunction,
        template: QubitTemplate,
        empty: Plaquette,
        idling_layer: PlaquetteLayer,
        stabilizing_layer: PlaquetteLayer,
        expansion_layer: PlaquetteLayer,
        slice_size: int,
    ) -> list[Block]:
        """Obtain the sequence of Block implementing a Magic State Cultivation.

        Returns:
            a sequence of Block representing a single attempt of the cultivation process.

        n.b. The cultivation process requires at least 3 Block and a target distance of at least 7.

        """
        static_circuit = CultivationBlock.__load_static_cultivation_circuit(orientation)
        static_subcircuits = CultivationBlock.__slice_into_subcircuits(static_circuit, slice_size)

        # fmt: off
        patch = DrawPolygon(
            {
                PauliBasis.Y: [
                    static_circuit.qubit_map.i2q[i]
                    for i in CultivationBlock.__CULTIVATION_PATCH
                ]
            }
        )
        # fmt: on

        static_layers_one = []
        static_layers_two = []
        for index, subcircuit in enumerate(static_subcircuits):
            plaquette = Plaquette(
                f"MSC_{orientation}_{index}",
                PlaquetteQubits([], sorted(subcircuit.qubits)),
                subcircuit,
                debug_information=PlaquetteDebugInformation(draw_polygons=patch),
            )

            layer = PlaquetteLayer(
                template,
                Plaquettes(FrozenDefaultDict({4: plaquette}, default_value=empty)),
            )

            if index < CultivationBlock.__BLOCK_ONE_SLICES:
                static_layers_one.append(layer)
            else:
                static_layers_two.append(layer)

        # Prepare Cultivation Block #1
        idling_layers = RepeatedLayer(idling_layer, height - (len(static_layers_one) - 2))
        block_one = Block(
            [
                static_layers_one[0],
                SequencedLayers([*static_layers_one[1:-1], idling_layers]),
                static_layers_one[-1],
            ]
        )

        # Prepare Cultivation Block #2
        stabilizing_layers = RepeatedLayer(stabilizing_layer, height - len(static_layers_two))
        block_two = Block(
            [
                static_layers_two[0],
                SequencedLayers([*static_layers_two[1:], expansion_layer, stabilizing_layers]),
                stabilizing_layer,
            ]
        )

        return [block_one, block_two]
