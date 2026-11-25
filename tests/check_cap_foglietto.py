"""Micro-check: le tracce scritte nella finestra 0
sopravvivono alle finestre successive?

MEM_CAP=256: il foglietto tiene le ultime 256 tracce.
Su episodi lunghi (16 finestre) le tracce dei mark
scritte all'inizio rischiano espulsione -> le query
tardive non potrebbero leggerli comunque vada il
canale. Misura quante tracce della finestra 0 restano.

python murlen/tests/check_cap_foglietto.py [cuda|cpu]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import murlen.model as MM  # noqa: E402
from murlen.ponte import vocab_da_sequenze  # noqa: E402
from universi_oltre_finestra import (  # noqa: E402
    K, TestaPredizione, sorpresa_simbolo)
from sonda_densita_n32 import (  # noqa: E402
    NFATTI, dati, finestre)

if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    seqs = dati(4, 7, NFATTI, 60)
    tutti = [list(range(0, 2000))]
    vocab, n = vocab_da_sequenze(
        seqs[:2].tolist() + tutti)
    torch.manual_seed(0)
    core = MM.MultiUniverse(n, K=K).to(dev)
    testa = TestaPredizione(
        max(vocab.values()) + 1,
        core.enc.tok.embedding_dim).to(dev)
    W = finestre(seqs, vocab, dev)
    z, mem = None, None
    m0 = None
    with torch.no_grad():
        for wi, (ids, pad) in enumerate(W):
            X, Y = ids[:, :-1], ids[:, 1:]
            z, h = core(X, pad[:, :-1], causal=True,
                        z0=z, mem0=mem)
            mem = core.annota(
                h, sorpresa_simbolo(testa(h), Y),
                pad[:, :-1])
            if wi == 0:
                m0 = [m.clone() for m in mem]
            vive = 0
            if m0 is not None:
                for b in range(len(mem)):
                    for r in m0[b]:
                        if torch.isclose(
                                mem[b], r).all(1).any():
                            vive += 1
            print('finestra %2d: tracce=%s  '
                  'finestra-0 vive: %d' % (
                      wi,
                      [m.shape[0] for m in mem],
                      vive), flush=True)
