from enum import Enum

import numpy
import stim

from tqec.circuit.measurement import Measurement
from tqec.circuit.moment import Moment
from tqec.circuit.qubit_map import QubitMap
from tqec.circuit.schedule import ScheduledCircuit
from tqec.compile.blocks.block import Block
from tqec.compile.blocks.cultivation.cultivation_circuit import CultivationCircuit
from tqec.compile.blocks.layers.atomic.base import BaseLayer
from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
from tqec.compile.blocks.layers.composed import RepeatedLayer, SequencedLayers
from tqec.plaquette import (
    FrozenDefaultDict,
    Plaquette,
    PlaquetteQubits,
    Plaquettes,
)
from tqec.templates.qubit import QubitTemplate
from tqec.utils import Orientation, TQECError
from tqec.utils.position import PhysicalQubitPosition2D, PlaquettePosition2D
from tqec.utils.scale import LinearFunction, PhysicalQubitScalable2D


class CultivationBlock:
    class Part(Enum):
        """Part identifier of the CultivationBlock."""

        ONE = 1
        TWO = 2

    MINIMAL_K_REQUIRED: int = 3
    """Smallest k (target distance 2k + 1) for which the static circuit fits in 2 blocks.
    Also required for the ExpansionTemplate, whose target distance must be strictly greater
    than ``CultivationCircuit.SOURCE_DISTANCE``."""

    def __init__(
        self,
        circuit: CultivationCircuit,
        plaquette_layers: tuple[PlaquetteLayer, ...],
        idling: PlaquetteLayer,
        plaquette_slot: int,
        height: LinearFunction,
        part: Part,
    ):
        """Initialize a CultivationBlock.

        Args:
            circuit: the contents of the static cultivation circuit
            plaquette_layers: the plaquette layers describing the layers of the cultivation circuit
            idling: a PlaquetteLayer representing idling that is used for padding
            plaquette_slot: the index of the plaquette where the cultivation occurs
            height: will determine how the layers are placed across the Blocks when materialized
            part: describes whether CultivationBlock is the first or second in the CultivationTower

        """
        self._circuit = circuit
        self._plaquette_layers = plaquette_layers
        self._idling = idling
        self._plaquette_slot = plaquette_slot
        self._height = height
        self._part = part

    @property
    def scalable_shape(self) -> PhysicalQubitScalable2D:
        """Shape of the surface code patch over which the cultivation is performed."""
        return self._idling.scalable_shape

    def observable_at(self, layer: BaseLayer | None, k: int) -> frozenset[Measurement]:
        """Return the measurements of ``layer`` to be included in the observable.

        Args:
            layer: a layer of the materialized block. Only the layers of the static circuit
                can contain measurements to be included in the observable.
            k: scaling factor.

        Returns:
            the measurements, with qubits in the coordinates of the block and offsets
            relative to the end of ``layer``.

        """
        index = next((i for i, own in enumerate(self._plaquette_layers) if own is layer), None)
        frame = self._circuit.observable_frame

        if index is not None and index in frame:
            anchor = self._cultivation_anchor(k)
            measurements = iter(m.offset_spatially_by(anchor.x, anchor.y) for m in frame[index])
        else:
            measurements = iter(())

        return frozenset(measurements)

    def _cultivation_anchor(self, k: int) -> PhysicalQubitPosition2D:
        """Position, in the coordinates of the block, where the cultivation circuit is anchored."""
        template = self._idling.template
        rows, cols = numpy.nonzero(template.instantiate(k) == self._plaquette_slot)
        slot = PlaquettePosition2D(int(cols[0]), int(rows[0]))
        return slot.get_origin_position(template.get_increments())

    @staticmethod
    def __idle_plaquette(schedule_size: int) -> Plaquette:
        return Plaquette(
            f"{CultivationCircuit.PREFIX}_IDLING",
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

        Returns:
            a tuple with the first and second `CultivationBlock` for a `CultivationTower`.

        """
        mirrored = mirrored_horizontal and orientation == Orientation.HORIZONTAL
        slot = 3 if mirrored else 4

        cultivation_circuit = CultivationCircuit(orientation, mirrored, schedule_size)

        plaquette_layers = tuple(
            CultivationBlock.__plaquette_layer(template, plaquette, empty, slot)
            for plaquette in cultivation_circuit.plaquettes
        )

        idling = CultivationBlock.__plaquette_layer(
            template, CultivationBlock.__idle_plaquette(schedule_size), empty, slot
        )

        block_one = CultivationBlock(
            cultivation_circuit, plaquette_layers, idling, slot, height, CultivationBlock.Part.ONE
        )
        block_two = CultivationBlock(
            cultivation_circuit, plaquette_layers, idling, slot, height, CultivationBlock.Part.TWO
        )

        return block_one, block_two

    def materialise(self, k: int) -> Block:
        """Turn this `CultivationBlock` into a `Block` based on the scaling factor k.

        The 11 layers of the static circuit are packed at the top of the lower two `Block`
        with the expansion layer as the first layer of the third `Block`.

        Args:
            k: the main parameter dictating which layers are organized in the Block.

        Raises:
            TQECError: if the cultivation does not fit in two blocks, i.e. for
                ``k < CultivationBlock.MINIMAL_K_REQUIRED``.

        Returns:
            the ``Block`` implementing this `CultivationBlock` at scale ``k``.

        """
        distance = 2 * k + 1

        if 2 * distance < len(self._plaquette_layers):
            raise TQECError(
                f"Cultivation needs {len(self._plaquette_layers)} rounds but two blocks only hold "
                f"{2 * distance} at k={k}; use k >= {CultivationBlock.MINIMAL_K_REQUIRED}."
            )

        split = max(0, len(self._plaquette_layers) - distance)

        match self._part:
            case CultivationBlock.Part.ONE:
                layers = self._plaquette_layers[:split]
            case CultivationBlock.Part.TWO:
                layers = self._plaquette_layers[split:]

        idle_count: int = distance - len(layers)
        start = self._idling if idle_count > 0 else layers[0]
        final = layers[-1] if layers else self._idling

        lower = 0 if idle_count > 0 else 1
        upper = max(len(layers) - 1, 0)
        middle = layers[lower:upper]
        filler = RepeatedLayer(self._idling, self._height - len(middle))

        return Block([start, SequencedLayers([filler, *middle]), final])
