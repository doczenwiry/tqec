import itertools
from enum import Enum
from pathlib import Path
from typing import ClassVar

import numpy
import stim

from tqec.circuit.moment import Moment
from tqec.circuit.qubit_map import QubitMap
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
from tqec.utils import Orientation, TQECError
from tqec.utils.scale import LinearFunction, PhysicalQubitScalable2D


def _batched(iterable, n):
    it = iter(iterable)
    while chunk := tuple(itertools.islice(it, n)):
        yield chunk


class CultivationBlock:
    class Part(Enum):
        """Part identifier of the CultivationBlock."""

        ONE = 1
        TWO = 2

    _STATIC_CIRCUIT: Path = Path(__file__).parent / "magic-state-cultivation-s-state.stim"
    _STEANE_ANCHOR: tuple[float, float] = (3.0, 1.0)
    _STEANE_CORNER: tuple[float, float] = (5.5, 7.5)

    # fmt: off
    _CULTIVATION_PATCH: ClassVar[list[int]] = [
        14, 22, 38, 39, 46, 47, 57, 63, 68, 64, 65, 69, 66, 67, 62,
        56, 45, 37, 29, 13,  5, 11,  4, 10,  9,  3,  0,  2,  1,  6,
    ]
    # fmt: on

    MINIMAL_K_REQUIRED: int = 3
    """Smallest k (target distance 2k + 1) for which the static circuit fits in 2 blocks."""

    def __init__(
        self,
        part: Part,
        height: LinearFunction,
        layers: tuple[PlaquetteLayer, ...],
        idling: PlaquetteLayer,
        expanding: PlaquetteLayer,
    ):
        """Initialize a CultivationBlock.

        Args:
            part: describes whether CultivationBlock is the first or second in the CultivationTower
            height: will determine how the layers are placed across the Blocks when materialised
            layers: the layers of the static cultivation circuit sliced beforehand
            idling: a PlaquetteLayer representing idling that is used for padding
            expanding: PlaquetteLayer representing the expansion of a surface code patch

        """
        self._part = part
        self._height = height
        self._layers = layers
        self._idling = idling
        self._expanding = expanding

    @property
    def scalable_shape(self) -> PhysicalQubitScalable2D:
        """Shape of the surface code patch over which the cultivation is performed."""
        return self._idling.scalable_shape

    @staticmethod
    def __idle_plaquette(schedule_size: int) -> Plaquette:
        return Plaquette(
            "IDLE_LAYER",
            PlaquetteQubits([], []),
            ScheduledCircuit(
                [Moment(stim.Circuit("SHIFT_COORDS"))],
                [schedule_size - 1],
                QubitMap(),
            ),
        )

    @staticmethod
    def __plaquette_layer(
        template: QubitTemplate, plaquette: Plaquette, empty: Plaquette, slot: int
    ) -> PlaquetteLayer:
        """Construct a plaquette layer from a given plaquette."""
        return PlaquetteLayer(
            template,
            Plaquettes(FrozenDefaultDict({slot: plaquette}, default_value=empty)),
        )

    @staticmethod
    def get_cultivation_blocks(
        orientation: Orientation,
        mirrored_horizontal: bool,
        height: LinearFunction,
        schedule_size: int,
        template: QubitTemplate,
        empty: Plaquette,
        expanding: PlaquetteLayer,
    ) -> tuple["CultivationBlock", "CultivationBlock"]:
        """Construct the pair of `CultivationBlock` that sit at the bottom of a `CultivationTower`.

        Args:
            orientation: the orientation of the walls of surface code patches
            mirrored_horizontal: whether HORIZONTAL surface code patches are mirrored (x -> -x),
                placing the cultivation in the bottom-left corner instead of the bottom-right.
                Ignored for VERTICAL patches, which are transposed instead.
            height: the number of rounds of stabilizer measurements (ignoring one layer for
                initialization and another for final measurement)
            schedule_size: the number of moments in one round under the convention being used
            template: a QubitTemplate of the memory patch over which cultivation is performed
            empty: an empty plaquette
            expanding: an PlaquetteLayer representing surface code expansion

        Returns:
            a tuple with the first and second `CultivationBlock` for a `CultivationTower`.

        """
        mirrored = mirrored_horizontal and orientation == Orientation.HORIZONTAL
        slot = 3 if mirrored else 4

        static_circuit = CultivationBlock.__load_static_cultivation_circuit(orientation, mirrored)
        static_layers = CultivationBlock.__slice_into_layers(
            static_circuit, orientation, mirrored, schedule_size, template, empty, slot
        )

        idling = CultivationBlock.__plaquette_layer(
            template, CultivationBlock.__idle_plaquette(schedule_size), empty, slot
        )

        block_one = CultivationBlock(
            CultivationBlock.Part.ONE, height, static_layers, idling, expanding
        )

        block_two = CultivationBlock(
            CultivationBlock.Part.TWO, height, static_layers, idling, expanding
        )

        return block_one, block_two

    def materialise(self, k: int) -> Block:
        """Turn this `CultivationBlock` into a `Block` based on the scaling factor k.

        The static circuit is sliced into one-round windows, followed by the expansion layer.
        Those layers are packed at the top of the two lower `Block` of the cultivation tower,
        so that it finishes exactly before the third `Block`.

        Args:
            k: the main parameter dictating which layers are organized in the Block.

        Raises:
            TQECError: if the cultivation does not fit in two blocks, i.e. for
                ``k < CultivationBlock.MINIMAL_K_REQUIRED``.

        Returns:
            the ``Block`` implementing this `CultivationBlock` at scale ``k``.

        """
        distance = 2 * k + 1
        all_layers = (*self._layers, self._expanding)

        if 2 * distance < len(all_layers):
            raise TQECError(
                f"Cultivation needs {len(all_layers)} rounds but two blocks only hold "
                f"{2 * distance} at k={k}; use k >= {CultivationBlock.MINIMAL_K_REQUIRED}."
            )

        split = max(0, len(all_layers) - distance)

        match self._part:
            case CultivationBlock.Part.ONE:
                layers = all_layers[:split]
            case CultivationBlock.Part.TWO:
                layers = all_layers[split:]

        idle_count: int = distance - len(layers)
        start = self._idling if idle_count > 0 else layers[0]
        final = layers[-1] if layers else self._idling

        lower = 0 if idle_count > 0 else 1
        upper = max(len(layers) - 1, 0)
        middle = layers[lower:upper]
        filler = RepeatedLayer(self._idling, self._height - len(middle))

        return Block([start, SequencedLayers([filler, *middle]), final])

    @staticmethod
    def __load_static_cultivation_circuit(
        orientation: Orientation,
        mirrored: bool,
    ) -> ScheduledCircuit:
        """Prepare the static circuit (Injection+Superdense+Doublecheck+Teleportation)."""
        anchor = numpy.array(CultivationBlock._STEANE_ANCHOR)
        corner = numpy.array(CultivationBlock._STEANE_CORNER)

        loaded = stim.Circuit.from_file(CultivationBlock._STATIC_CIRCUIT)
        static = stim.Circuit()

        for instruction in loaded.flattened():
            assert isinstance(instruction, stim.CircuitInstruction)
            if instruction.name == "QUBIT_COORDS":
                location = numpy.array(instruction.gate_args_copy())
                position = 2 * (location - (anchor + corner))
                if orientation == Orientation.VERTICAL:
                    position[0], position[1] = position[1], position[0]
                elif mirrored:
                    position[0] = -position[0]
                static.append("QUBIT_COORDS", instruction.targets_copy(), position)
            elif instruction.name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                static.append(instruction)

        return ScheduledCircuit.from_circuit(static)

    @staticmethod
    def __slice_into_layers(
        circuit: ScheduledCircuit,
        orientation: Orientation,
        mirrored: bool,
        schedule_size: int,
        template: QubitTemplate,
        empty: Plaquette,
        slot: int,
    ) -> tuple[PlaquetteLayer, ...]:
        """Slice a circuit into pieces of the required size."""
        label = ("M" if mirrored else "") + orientation.name[0]
        patch = DrawPolygon(
            {PauliBasis.Y: [circuit.qubit_map.i2q[i] for i in CultivationBlock._CULTIVATION_PATCH]}
        )

        layers: list[PlaquetteLayer] = []
        for index, moments in enumerate(_batched(circuit.scheduled_moments, schedule_size)):
            subcircuit = ScheduledCircuit(
                [m for _, m in moments], list(range(schedule_size)), circuit.qubit_map
            )
            plaquette = Plaquette(
                f"MSC_{label}_{index}",
                PlaquetteQubits([], sorted(subcircuit.qubits)),
                subcircuit,
                debug_information=PlaquetteDebugInformation(draw_polygons=patch),
            )

            layers.append(CultivationBlock.__plaquette_layer(template, plaquette, empty, slot))

        return tuple(layers)
