"""A quanti fatti nasce il lettore?

sonda_lettore: nemmeno con il curriculum (4->60 fatti, 900
passi) il canale foglietto->predizione si forma — ablazione
intatto/svuotato/episodio-sbagliato identiche. Pero' su
dialogo/catena/gate il foglietto VIENE letto: la' i fatti
usati erano 1-2 e servivano dentro il flusso.

Discriminante: lo STESSO compito a pochissimi fatti.
Mente fresca per livello, 300 passi, poi l'ablazione.

- se intatto < svuotato a 1-2 fatti: il lettore esiste ma
  non scala a 60 tracce — il problema e' il NUMERO
- se resta piatto anche a 1: il problema e' la struttura
  "scrivi poi esame" — il lettore nasce solo con l'uso
  in-linea, e memcap misura il vuoto per sempre

python murlen/tests/sonda_lettore_min.py [cuda|cpu]"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import murlen.model as MM  # noqa: E402
from murlen.ponte import (vocab_da_sequenze, tensorizza)  # noqa: E402
from universi_oltre_finestra import allena  # noqa: E402
import sonda_memcap as SM  # noqa: E402
from sonda_memcap import (dati, lettore, seqs_unite,  # noqa: E402
                          CAMPO, K)

PASSI = 300
LIVELLI = [1, 2, 4, 8]


def finestre(seqs, vocab, dev):
    out = []
    for a in range(0, len(seqs[0]), CAMPO):
        ids, pad = tensorizza(
            [s[a:a + CAMPO] for s in seqs], vocab, CAMPO)
        out.append((ids.to(dev), pad.to(dev)))
    return out


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    gate = len(sys.argv) > 2 and sys.argv[2] == 'gate'
    print('scrittura: %s' % ('gate appreso' if gate
                             else 'fence sorpresa'))
    for nf in LIVELLI:
        scr_t, ric_t, _ = dati(PASSI, 1, nf)
        scr_v, ric_v, _ = dati(8, 2, nf)
        vocab, n = vocab_da_sequenze(
            seqs_unite(scr_t, ric_t) + seqs_unite(scr_v, ric_v))
        torch.manual_seed(0)
        core = MM.MultiUniverse(n, K=K, d=192, ctx=CAMPO,
                                enc_layers=4, m=16, nhead=4,
                                tempo=64, mem_gate=gate)
        hist, testa = allena(
            core, finestre(seqs_unite(scr_t, ric_t), vocab,
                           dev),
            vocab, dev, passi=PASSI)
        print('=== %d fatti — train %.3f -> %.3f ===' % (
            nf, hist[0], sum(hist[-10:]) / 10), flush=True)
        # lettore() usa SM.N_FATTI per le posizioni delle
        # risposte: lo allineo al livello corrente
        SM.N_FATTI = nf
        lettore(core, testa, scr_v, ric_v, vocab, dev)
