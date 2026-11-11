from __future__ import annotations

from typing import TYPE_CHECKING

from tqec.circuit.measurement import Measurement
from tqec.circuit.measurement_map import MeasurementRecordsMap
from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.blocks.positioning import LayoutPosition2D
from tqec.compile.observables.abstract_observable import AbstractObservable
from tqec.compile.observables.builder import (
    Observable,
    ObservableBuilder,
    ObservableComponent,
    get_observable_with_measurement_records,
)
from tqec.compile.specs.library.generators.expansion import SurfaceCodeExpansionLayer
from tqec.utils.position import BlockPosition2D

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tqec.compile.blocks.cultivation.cultivation_block import CultivationBlock
    from tqec.compile.tree.node import LayerNode


def get_ordered_leaves(root: LayerNode) -> list[LayerNode]:
    """Return the leaves of the tree in time order."""
    if root.is_leaf:
        return [root]
    return [n for child in root.children for n in get_ordered_leaves(child)]


def annotate_observable(
    root: LayerNode,
    k: int,
    observable: AbstractObservable,
    observable_index: int,
    observable_builder: ObservableBuilder,
    slices_with_temporal_hadamard_layer: set[int],
    cultivation_blocks: Mapping[int, Mapping[LayoutPosition2D, CultivationBlock]] | None = None,
) -> None:
    """Annotates the observables on the tree.

    Args:
        root: root node of the tree.
        k: distance parameter.
        observable: observable to annotate.
        observable_index: index of the observable in the circuit.
        observable_builder: builder that computes and constructs qubits whose
            measurements will be included in the logical observable.
        slices_with_temporal_hadamard_layer: z slices containing a temporal
            Hadamard layer.
        cultivation_blocks: the cultivation blocks in the computation by z-slice and position

    """
    blocks_by_z = cultivation_blocks or {}
    for z, subtree_root in enumerate(root.children):
        leaves = get_ordered_leaves(subtree_root)
        obs_slice = observable.slice_at_z(z)

        # Annotate the observable at the bottom of the blocks
        _annotate_observable_at_node(
            leaves[0],
            obs_slice,
            k,
            observable_index,
            observable_builder,
            ObservableComponent.BOTTOM_STABILIZERS,
        )

        readout_layer = leaves[-1]
        if z in slices_with_temporal_hadamard_layer:
            readout_layer = leaves[-2]

            if obs_slice.temporal_hadamard_pipes:
                _annotate_observable_at_node(
                    leaves[-1],
                    obs_slice,
                    k,
                    observable_index,
                    observable_builder,
                    ObservableComponent.REALIGNMENT,
                )

        # Annotate the observable at the top of the blocks
        _annotate_observable_at_node(
            readout_layer,
            obs_slice,
            k,
            observable_index,
            observable_builder,
            ObservableComponent.TOP_READOUTS,
        )

        # Annotate the observable in the cultivation towers
        sorted_towers = sorted(obs_slice.cultivation_towers, key=lambda t: t.position.as_tuple())
        for tower in sorted_towers:
            block_position = BlockPosition2D(tower.position.x, tower.position.y)
            position = LayoutPosition2D.from_block_position(block_position)
            block = blocks_by_z.get(z, {}).get(position)
            if block is not None:
                for leaf in leaves:
                    _annotate_cultivation_frame_at_node(
                        leaf,
                        position,
                        block,
                        block_position,
                        k,
                        observable_index,
                        ObservableComponent.CULTIVATION,
                    )
            else:
                # Without a block in this slice, the slice is the top of the tower.
                below = blocks_by_z.get(z - 1, {}).get(position)
                if below is not None:
                    _annotate_expansion_at_node(
                        leaves[0],
                        position,
                        below,
                        block_position,
                        k,
                        observable_index,
                        ObservableComponent.EXPANSION,
                    )


def _annotate_observable_at_node(
    node: LayerNode,
    obs_slice: AbstractObservable,
    k: int,
    observable_index: int,
    observable_builder: ObservableBuilder,
    component: ObservableComponent,
) -> None:
    circuit = node.get_annotations(k).circuit
    assert circuit is not None
    measurement_record = MeasurementRecordsMap.from_scheduled_circuit(circuit)
    assert isinstance(node._layer, LayoutLayer)
    template, _ = node._layer.to_template_and_plaquettes()
    obs_qubits = observable_builder.build(k, template, obs_slice, component)
    if obs_qubits:
        obs_annotation = get_observable_with_measurement_records(
            obs_qubits, measurement_record, observable_index
        )
        node.get_annotations(k).observables.append(obs_annotation)


def _annotate_cultivation_frame_at_node(
    node: LayerNode,
    position: LayoutPosition2D,
    block: CultivationBlock,
    block_position: BlockPosition2D,
    k: int,
    observable_index: int,
    component: ObservableComponent,
) -> None:
    """Include the frame measurements of ``block`` performed by ``node`` in the observable."""
    assert component == ObservableComponent.CULTIVATION
    if isinstance(node._layer, LayoutLayer):
        measurements = block.observable_at(node._layer.layers.get(position), k)
        _annotate_block_measurements_at_node(
            node, measurements, block_position, k, observable_index
        )


def _annotate_block_measurements_at_node(
    node: LayerNode,
    measurements: frozenset[Measurement],
    block_position: BlockPosition2D,
    k: int,
    observable_index: int,
) -> None:
    """Include ``measurements``, in the coordinates of a block, in the observable."""
    if measurements:
        assert isinstance(node._layer, LayoutLayer)
        template, _ = node._layer.to_template_and_plaquettes()
        shape = template.element_shape(k)
        increments = template.get_increments()
        dx = block_position.x * shape.x * increments.x
        dy = block_position.y * shape.y * increments.y

        circuit = node.get_annotations(k).circuit
        assert circuit is not None
        records = MeasurementRecordsMap.from_scheduled_circuit(circuit)

        shifted = [m.offset_spatially_by(dx, dy) for m in measurements]
        node.get_annotations(k).observables.append(
            Observable(
                observable_index,
                [m.qubit for m in shifted],
                [records[m.qubit][m.offset] for m in shifted],
            )
        )


def _annotate_expansion_at_node(
    node: LayerNode,
    position: LayoutPosition2D,
    block: CultivationBlock,
    block_position: BlockPosition2D,
    k: int,
    observable_index: int,
    component: ObservableComponent,
) -> None:
    """Include the expansion correction of ``block`` performed by ``node`` in the observable."""
    assert component == ObservableComponent.EXPANSION
    if isinstance(node._layer, LayoutLayer):
        layer = node._layer.layers.get(position)
        if isinstance(layer, SurfaceCodeExpansionLayer):
            measurements = layer.observable_correction(block.observable_representative(k), k)
            _annotate_block_measurements_at_node(
                node, measurements, block_position, k, observable_index
            )
