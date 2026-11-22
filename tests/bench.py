"""Misura il tempo per passo di impara_predizione.
python murlen/tests/bench.py [cpu|cuda]  (dalla cartella sopra murlen)"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import vocab_da_sequenze, impara_predizione  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def misura(K, tempo, passi=2, **kw):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    seqs = [torch.randint(0, 200, (int(n),), generator=g).tolist()
            for n in torch.randint(40, 200, (16,), generator=g)]
    vocab, n = vocab_da_sequenze(seqs)
    core = MultiUniverse(n, K=K, comm=K > 1, tempo=tempo)
    impara_predizione(core, seqs, vocab, passi=1, **kw)  # riscaldamento
    sync()
    t = time.perf_counter()
    hist, _ = impara_predizione(core, seqs, vocab, passi=passi, **kw)
    sync()
    dt = (time.perf_counter() - t) / passi
    print('K=%d tempo=%d %s: %.2f s/passo, loss %.3f -> %.3f'
          % (K, tempo, kw or '', dt, hist[0], hist[-1]))


if __name__ == '__main__':
    torch.set_num_threads(4)
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    for K, tempo, kw in ((1, 0, {}), (3, 0, {}), (3, 64, {}),
                         (3, 0, dict(batch=4)),
                         (3, 64, dict(batch=4))):
        try:
            misura(K, tempo, device=dev, **kw)
        except torch.OutOfMemoryError:
            print('K=%d tempo=%d %s: memoria GPU esaurita'
                  % (K, tempo, kw))
        torch.cuda.empty_cache()
