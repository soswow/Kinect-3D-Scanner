"""Final-only weighted block allocation; voxel update equations are unchanged."""

import sys
import open3d.core as o3c


def activate_fusion_blocks(engine, hashmap, blocks):
    """Keep ordinary activation; private weighted Final activates missing keys.

    Native Activate reserves Size()+input rows, including existing keys. The
    Final planner chooses an exact capacity, so redundant keys must not enter
    Activate. Reviewed callers still perform their original full-frustum find
    and integrate every original voxel after this function returns.
    """
    if not getattr(engine, "_final_missing_only_activation", False):
        return hashmap.activate(blocks)
    capacity = engine._final_allocated_blocks
    logical_limit = engine._fusion_block_limit
    if (engine.settings.confidence_fusion is not True or type(capacity) is not int
            or type(logical_limit) is not int or not 1 <= capacity <= logical_limit):
        raise ValueError("Invalid private weighted Final allocation policy")
    native_output = missing = None
    try:
        before_size = int(hashmap.size())
        if int(hashmap.capacity()) != capacity or not 0 <= before_size <= capacity:
            raise ValueError("Weighted Final capacity changed before activation")
        _, found = hashmap.find(blocks)
        missing = blocks[found.logical_not()]
        added = len(missing)
        if before_size + added > capacity:
            raise ValueError("Weighted Final keys exceed their exact planned allocation")
        if added:
            native_output = hashmap.activate(missing)
        if (int(hashmap.capacity()) != capacity
                or int(hashmap.size()) != before_size + added):
            raise ValueError("Weighted Final activation changed the planned capacity or key count")
        _, complete = hashmap.find(blocks)
        if not complete.cpu().numpy().all():
            raise ValueError("Weighted Final activation omitted original frustum keys")
    finally:
        # Retain missing keys and native output until selected-device work is
        # complete, even if native activation partly wrote then raised. Final's
        # private candidate is discarded on failure; there is no retry.
        if str(engine.device).startswith("CUDA:"):
            primary = sys.exc_info()[1]
            try:
                o3c.cuda.synchronize(engine.device)
            except BaseException as cleanup:
                if primary is not None:
                    add_note = getattr(primary, "add_note", None)
                    if callable(add_note):
                        add_note(f"Final activation synchronization also failed: {cleanup}")
                    raise primary from cleanup
                raise
