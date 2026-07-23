"""Background-prefetching training loader.

Experiment 04 generated each batch synchronously, inline, right before the
forward pass -- so the (single-threaded, pure-Python) data generation and the
GPU compute ran serially and the H100 would sit idle ~18-50ms every step waiting
for Python. This runs the generator in a separate process that tensorises batches
and pushes them onto a bounded queue, so generation overlaps with GPU compute and
the GPU never waits.

Determinism: with num_workers=1 there is a single seeded generator and batch order
is exactly reproducible (identical to calling stream_training_batches directly).
num_workers>1 trades exact order-determinism for throughput (each worker has its
own seed offset); the data *distribution* stays seeded either way. We default to 1
since measured single-core gen (~14k ex/s) already outpaces one H100's consumption
at our batch sizes.
"""

import torch
import torch.multiprocessing as mp

from data_gen import stream_training_batches


def _worker(seed, held_out_cells, batch_size, queue, stop):
    stream = stream_training_batches(seed, held_out_cells, batch_size)
    while not stop.is_set():
        batch, k, n = next(stream)
        input_ids = torch.tensor([ex[0] for ex in batch], dtype=torch.long)
        queue.put((input_ids, k, n))  # blocks when queue full -> natural backpressure


class PrefetchTrainLoader:
    def __init__(self, seed, held_out_cells, batch_size, num_workers=1, queue_size=8):
        self.ctx = mp.get_context("spawn")
        self.queue = self.ctx.Queue(maxsize=queue_size)
        self.stop = self.ctx.Event()
        self.procs = []
        for w in range(num_workers):
            p = self.ctx.Process(
                target=_worker,
                args=(seed + w * 100_003, held_out_cells, batch_size, self.queue, self.stop),
                daemon=True,
            )
            p.start()
            self.procs.append(p)

    def __iter__(self):
        return self

    def __next__(self):
        return self.queue.get()

    def close(self):
        self.stop.set()
        # drain so workers blocked on put() can exit
        try:
            while True:
                self.queue.get_nowait()
        except Exception:
            pass
        for p in self.procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
