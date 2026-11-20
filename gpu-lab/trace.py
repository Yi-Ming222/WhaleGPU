"""CPU trace shared by the command line and browser experiments."""


def thread_indices(n, blocks, threads, step=None):
    stride = blocks * threads if step is None else step
    return [list(range(tid, n, stride)) for tid in range(blocks * threads)]


def reduction_rounds(values, pairing="halves"):
    """Yield independent shared memory snapshots and per-lane operations."""
    shared = list(values)
    gap = len(shared) // 2 if pairing == "halves" else 1
    while 0 < gap < len(shared):
        lanes = range(gap) if pairing == "halves" else range(0, len(shared), 2 * gap)
        operations = []
        for lane in lanes:
            left, right = shared[lane], shared[lane + gap]
            shared[lane] = left + right
            operations.append(dict(lane=lane, partner=lane + gap, left=left,
                                   right=right, result=shared[lane]))
        yield gap, shared.copy(), operations
        gap = gap // 2 if pairing == "halves" else gap * 2


def main():
    xs = list(range(1, 14))
    blocks, threads = 2, 4
    assignments = thread_indices(len(xs), blocks, threads)
    print("CPU trace. Input:", xs)
    totals = []
    for block in range(blocks):
        shared = []
        for lane in range(threads):
            indices = assignments[block * threads + lane]
            subtotal = sum(xs[i] for i in indices)
            print(f"block={block} lane={lane} indices={indices} subtotal={subtotal}")
            shared.append(subtotal)
        print("  after all lanes write:", shared)
        for width, shared, _ in reduction_rounds(shared):
            print(f"  width={width}, active values={shared[:width]}")
        totals.append(shared[0])
    print("Block totals:", totals, "| final:", sum(totals), "| reference:", sum(xs))


if __name__ == "__main__":
    main()
