import itertools
from pathlib import Path
from typing import ClassVar

import numpy
import stim

from tqec.circuit.measurement import Measurement
from tqec.circuit.schedule import ScheduledCircuit
from tqec.plaquette import Plaquette, PlaquetteQubits
from tqec.plaquette.debug import DrawPolygon, PlaquetteDebugInformation
from tqec.plaquette.rpng import PauliBasis
from tqec.utils import Orientation, TQECError


def _batched(iterable, n):
    it = iter(iterable)
    while chunk := tuple(itertools.islice(it, n)):
        yield chunk


class CultivationCircuit:
    PREFIX: str = "MSC"

    _STATIC_CIRCUIT: Path = Path(__file__).parent / "magic-state-cultivation-s-state.stim"
    """The path to the static cultivation circuit to be loaded.

    Several assumptions are made about it, some of which are checked at loading-time:
    - It contains the five stages; injection+superdense+doublecheck+teleportation+destruction
    - It is made up of 77 moments in total for a rounded slicing with the 7-moments QEC schedules
    - It contains a complete set of DETECTOR annotations
    - It contains one OBSERVABLE_INCLUDE annotation describing the frame for tracking
    - It contains one OBSERVABLE_INCLUDE annotation providing the representative as an MPP
    - The final two lines are the representative observable and will be discarded at loading-time
    """
    _STATIC_MOMENTS: int = 77
    """The number of moments the static circuit should contain"""
    _STEANE_ANCHOR: tuple[float, float] = (3.0, 1.0)
    """The location where the Steane Code patch (i.e. d=3 color code) is anchored"""
    _STEANE_CORNER: tuple[float, float] = (5.5, 7.5)

    # fmt: off
    _PATCH: ClassVar[list[int]] = [
        14, 22, 38, 39, 46, 47, 57, 63, 68, 64, 65, 69, 66, 67, 62,
        56, 45, 37, 29, 13,  5, 11,  4, 10,  9,  3,  0,  2,  1,  6,
    ]
    """The qubits selected to draw the patch as a polygon in the generated circuit"""
    # fmt: on

    def __init__(self, orientation: Orientation, mirrored: bool, schedule_size: int):
        """Prepare the sliced static cultivation circuit.

        Args:
            orientation: describes the color of the patches boundaries in the XZ and YZ planes.
            mirrored: indicates whether the patches are mirrored or not.
            schedule_size: the size of the schedule for the slicing.

        """
        self._orientation = orientation
        self._mirrored = mirrored
        self._schedule_size = schedule_size

        anchor = numpy.array(CultivationCircuit._STEANE_ANCHOR)
        corner = numpy.array(CultivationCircuit._STEANE_CORNER)

        loaded = stim.Circuit.from_file(CultivationCircuit._STATIC_CIRCUIT)

        # Prepare the static circuit (Injection+Superdense+Doublecheck+Teleportation+Destruction).
        static = stim.Circuit()
        records: list[tuple[int, int]] = []  # (moment, qubit) of every measurement
        observable_include: set[int] = set()  # Indices of records to include in the observable
        moment: int = 0
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
            elif instruction.name == "MPP":
                # Stop processing the circuit
                break
            elif instruction.name == "DETECTOR":
                continue  # Ignore the DETECTOR annotations for now.
            elif instruction.name == "OBSERVABLE_INCLUDE":
                if instruction.gate_args_copy() != [0]:
                    raise TQECError(
                        f"Expected OBSERVABLE_INCLUDE(0) in {CultivationCircuit._STATIC_CIRCUIT.name}, "  # noqa: E501
                        f"found OBSERVABLE_INCLUDE({instruction.gate_args_copy()})."
                    )
                if len(observable_include) != 0:
                    raise TQECError("Loaded circuit should contain a single OBSERVABLE_INCLUDE.")
                observable_include.update(
                    len(records) + target.value for target in instruction.targets_copy()
                )
            else:
                if instruction.name == "TICK":
                    moment += 1
                elif stim.gate_data(instruction.name).produces_measurements:
                    records.extend((moment, target.value) for target in instruction.targets_copy())
                static.append(instruction)

        if not observable_include:
            raise TQECError(
                f"No OBSERVABLE_INCLUDE found in {CultivationCircuit._STATIC_CIRCUIT.name}."
            )

        # Validate the number of moments in the ScheduledCircuit.
        self._circuit = ScheduledCircuit.from_circuit(static)
        moment_count = sum(1 for _ in self._circuit.scheduled_moments)
        if moment_count != CultivationCircuit._STATIC_MOMENTS:
            raise TQECError(
                f"Expected {CultivationCircuit._STATIC_MOMENTS} moments in "
                f"{CultivationCircuit._STATIC_CIRCUIT.name}, found {moment_count}."
            )

        # Convert the measurements to be included in the observable into a frame.
        frame: dict[int, set[Measurement]] = {}
        for index in sorted(observable_include):
            moment, qubit = records[index]
            layer = moment // schedule_size
            later = sum(
                1 for m, q in records[index + 1 :] if q == qubit and m // schedule_size == layer
            )
            measurement = Measurement(self._circuit.qubit_map.i2q[qubit], -(later + 1))
            frame.setdefault(layer, set()).add(measurement)
        self._observable_frame = {w: frozenset(ms) for w, ms in frame.items()}

        self._plaquettes = CultivationCircuit.__slice_into_plaquettes(
            self._circuit, orientation, mirrored, schedule_size
        )

    @property
    def observable_frame(self) -> dict[int, frozenset[Measurement]]:
        """Return the frame of measurements to be included in an observable."""
        return dict(self._observable_frame)

    @property
    def plaquettes(self) -> tuple[Plaquette, ...]:
        """Return the plaquettes that constitute this circuit."""
        return self._plaquettes

    @staticmethod
    def __slice_into_plaquettes(
        circuit: ScheduledCircuit, orientation: Orientation, mirrored: bool, schedule_size: int
    ) -> tuple[Plaquette, ...]:
        """Slice a circuit into pieces of the required size."""
        moment_count = sum(1 for _ in circuit.scheduled_moments)
        if moment_count % schedule_size != 0:
            raise TQECError(
                f"The static cultivation circuit ({moment_count} moments) cannot be sliced into "
                f"whole rounds of {schedule_size} moments."
            )

        label = ("M" if mirrored else "") + orientation.name[0]
        patch = DrawPolygon(
            {PauliBasis.Y: [circuit.qubit_map.i2q[i] for i in CultivationCircuit._PATCH]}
        )

        plaquettes: list[Plaquette] = []
        for index, moments in enumerate(_batched(circuit.scheduled_moments, schedule_size)):
            subcircuit = ScheduledCircuit(
                [m for _, m in moments], list(range(schedule_size)), circuit.qubit_map
            )
            plaquettes.append(
                Plaquette(
                    f"{CultivationCircuit.PREFIX}_{label}_{index}",
                    PlaquetteQubits([], sorted(subcircuit.qubits)),
                    subcircuit,
                    debug_information=PlaquetteDebugInformation(draw_polygons=patch),
                )
            )

        return tuple(plaquettes)
