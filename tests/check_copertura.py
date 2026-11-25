"""Copertura di scrittura: quanti fatti lasciano traccia?

Il fence sceglie gli outlier di sorpresa DENTRO la
finestra (soglia relativa). Se la finestra e'
uniformemente nuova la soglia si alza e pochi fatti
entrano nel foglietto: per gli altri non esiste
niente da leggere — la sonda lettore non puo'
accendersi comunque vada il canale.

Qui: per ogni fatto, conto se una sua posizione
(mark/ans o una sua query/giun) e' stata scritta
come traccia, finestra per finestra.

python murlen/tests/check_copertura.py [cuda|cpu]
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
from sonda_pressione_indirizzo import WIN  # noqa: E402


def fatto_di(g, nfatti):
    """posizione globale g -> fatto a cui appartiene."""
    nm = 2 * nfatti
    if g < nm:
        return g // 2
    return ((g - nm) // 2) % nfatti


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
    coperti = [set() for _ in range(seqs.shape[0])]
    with torch.no_grad():
        for wi, (ids, pad) in enumerate(W):
            X, Y = ids[:, :-1], ids[:, 1:]
            z, h = core(X, pad[:, :-1], causal=True,
                        z0=z, mem0=mem)
            lo = testa(h)
            s = sorpresa_simbolo(lo, Y)
            sel = core.foglio.scegli_valori(s,
                                            pad[:, :-1])
            mem = core.annota(h, s, pad[:, :-1])
            nuovi = [0] * seqs.shape[0]
            for b in range(seqs.shape[0]):
                for p in sel[b].nonzero(
                        ).flatten().tolist():
                    f = fatto_di(wi * WIN + p, NFATTI)
                    if f not in coperti[b]:
                        coperti[b].add(f)
                        nuovi[b] += 1
            print('finestra %2d: tracce scritte=%s  '
                  'fatti nuovi coperti=%s  '
                  'copertura=%s/32' % (
                      wi,
                      [int(sel[b].sum()) for b in
                       range(seqs.shape[0])],
                      nuovi,
                      [len(c) for c in coperti]),
                  flush=True)
