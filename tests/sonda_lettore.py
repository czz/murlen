"""Il lettore del foglietto nasce con un curriculum?

sonda_memcap ha misurato che su 300 passi diretti il canale
foglietto->predizione non esiste (ablazione intatto /
svuotato / episodio-1 identiche). Qui lo stesso compito a
stadi crescenti: pochi fatti per episodio prima, molti poi
— il lettore si forma su esercizi alla portata e scala.

Se intatto < svuotato alla fine, l'organo funziona quando
allenato e il cap del foglietto torna misurabile sul costo;
se restano uguali, il limite non era il curriculum.

Strumento di laboratorio: il curriculum qui e' uno stampo
della sonda — verifica che l'organo *possa* funzionare, non
che la mente lo scopra da sola.

python murlen/tests/sonda_lettore.py [cuda|cpu]"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import murlen.model as MM  # noqa: E402
from murlen.ponte import (vocab_da_sequenze, tensorizza)  # noqa: E402
from universi_oltre_finestra import allena  # noqa: E402
from sonda_memcap import (dati, misura, lettore,  # noqa: E402
                          seqs_unite, CAMPO, K)

# (fatti per episodio, passi) — crescente fino al carico pieno
STADI = [(4, 200), (12, 200), (30, 200), (60, 300)]


def finestre(seqs, vocab, dev):
    """Le sequenze scrittura+richiamo spezzate in finestre da
    CAMPO simboli per l'allenamento a blocchi."""
    out = []
    lun = len(seqs[0])
    for a in range(0, lun, CAMPO):
        ids, pad = tensorizza(
            [s[a:a + CAMPO] for s in seqs], vocab, CAMPO)
        out.append((ids.to(dev), pad.to(dev)))
    return out


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    # vocabolario su tutti gli stadi + la valutazione
    tutte = []
    for st, (nf, _) in enumerate(STADI):
        a, b, _ = dati(2, 100 + st, nf)
        tutte += seqs_unite(a, b)
    scr_v, ric_v, etas = dati(8, 2)
    tutte += seqs_unite(scr_v, ric_v)
    vocab, n = vocab_da_sequenze(tutte)

    torch.manual_seed(0)
    core = MM.MultiUniverse(n, K=K, d=192, ctx=CAMPO,
                            enc_layers=4, m=16, nhead=4,
                            tempo=64)
    testa = None
    for st, (nf, passi) in enumerate(STADI):
        scr, ric, _ = dati(passi, 1 + st, nf)
        hist, testa = allena(
            core, finestre(seqs_unite(scr, ric), vocab, dev),
            vocab, dev, passi=passi, testa=testa)
        print('stadio %d: %d fatti, train %.3f -> %.3f' % (
            st, nf, hist[0], sum(hist[-10:]) / 10),
            flush=True)

    lettore(core, testa, scr_v, ric_v, vocab, dev)
    for cap in (8, 32, 64, 256):
        misura(core, testa, scr_v, ric_v, etas, vocab, dev,
               cap)
