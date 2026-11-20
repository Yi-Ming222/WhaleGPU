"""CPU exercises. Use the checks in check.py as your target."""


def map_double(xs):
    # Worked example: one output per input.
    return [2 * x for x in xs]


def zip_add(xs, ys):
    # TODO 1: add corresponding values. Raise ValueError for different lengths.
    raise NotImplementedError


def grid_stride_indices(n, blocks, threads):
    # TODO 2: return one list per global thread, in global thread order.
    raise NotImplementedError


def shared_neighbor_sums(xs, threads):
    # TODO 3: stage a tile, then sum each lane with its right neighbor.
    # The neighbor contributes zero at a block boundary or past the input.
    raise NotImplementedError


def block_sums(xs, blocks, threads):
    # TODO 4: grid-stride accumulation, then pairwise halving per block.
    # Return one subtotal per block. Assume threads is a positive power of two.
    raise NotImplementedError
