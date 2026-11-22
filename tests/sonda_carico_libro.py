"""Nei libri veri esiste il carico che allena il lettore?

sonda_lettore_fattori + densita: il lettore nasce quando lo
stesso fatto e' letto >=4 volte E la traccia e' l'unica via
alla predizione. sonda_lettore_libro (piatto, 2.935~2.934)
non distingueva pero' due casi:

- riuso DETERMINISTICO: il contesto e' riapparso e la
  continuazione e' la STESSA della prima volta -> la
  memoria puo' essere load-bearing: e' una "lettura"
  potenziale del canale
- riuso AMBIGUO: il contesto si ripete ma continua
  diversamente -> la traccia non aiuta: rumore che puo'
  aver diluito un segnale piccolo

Qui si separano: per pezzo, quante posizioni di ciascun
tipo; quante volte lo stesso contesto si ripete (le
"letture per fatto" da confrontare col ginocchio ~4); e la
CE per tipo x condizione (intatto/svuotato/altro). Se la
CE intatta scende SOLO sul deterministico, il lettore ha un
segnale vero su testo naturale — anche se la media resta
piatta.

python murlen/tests/sonda_carico_libro.py [cuda|cpu]"""
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import (vocab_da_sequenze, PAD_ID)  # noqa: E402
from universi_oltre_finestra import (allena,  # noqa: E402
                                     sorpresa_simbolo)
from sonda_campo_libro import pezzi_liberi, CAMPO  # noqa: E402
from sonda_lettore_libro import finestre  # noqa: E402
from universi_libro import libri  # noqa: E402

PASSI, K, G = 200, 2, 16


def maschere_riuso(seqs):
    """Per posizione q: (riuso, deterministico). Riuso = il
    contesto G prima di q e' gia' apparso; deterministico =
    la sua prima occorrenza era seguita dallo STESSO
    simbolo di q. Conta anche le ripetizioni per contesto.
    -> maschere, ripetizioni per contesto"""
    out, ripetizioni = [], []
    for s in seqs:
        visti = {}
        riu, det = [False] * len(s), [False] * len(s)
        for q in range(G, len(s)):
            c = tuple(s[q - G:q])
            if c in visti:
                riu[q] = True
                det[q] = visti[c][0] == s[q]
                visti[c][1] += 1
            else:
                visti[c] = [s[q], 1]
        out.append((riu, det))
        ripetizioni += [v[1] for v in visti.values()
                        if v[1] > 1]
    return out, ripetizioni


@torch.no_grad()
def ce_riuso(core, testa, W, maschere, stati, dev,
             mem_alt, det_solo):
    """CE media alle posizioni di riuso (det_solo: solo le
    deterministiche) per finestre >=1, mem0 per condizione."""
    tot, n = 0., 0
    for wi, (ids, pad) in enumerate(W):
        if wi == 0:
            continue
        X, Y = ids[:, :-1].to(dev), ids[:, 1:].to(dev)
        z_in, mem_in = stati[wi]
        if mem_alt == 'svuotato':
            mem0 = None
        elif mem_alt == 'altro':
            mem0 = [mem_in[(b + 1) % len(mem_in)]
                    for b in range(len(mem_in))]
        else:
            mem0 = mem_in
        _, h = core(X, pad[:, :-1].to(dev), causal=True,
                    z0=z_in, mem0=mem0)
        ce = nn.functional.cross_entropy(
            testa(h).flatten(0, 1), Y.flatten(),
            reduction='none', ignore_index=PAD_ID
            ).view(Y.shape)
        a = wi * CAMPO
        for b in range(ids.size(0)):
            for j in range(Y.size(1)):
                q = a + j + 1
                if q < len(maschere[b][0]) and \
                        maschere[b][0][q] and \
                        (not det_solo or maschere[b][1][q]):
                    tot += ce[b, j].item()
                    n += 1
    return tot / max(n, 1), n


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    r = random.Random(1)
    testi = libri(per_lingua=200)
    train, test = pezzi_liberi(testi, r, 1200, 32)
    print('%d pezzi train, %d test' % (len(train), len(test)))
    vocab, n = vocab_da_sequenze(train + test)
    torch.manual_seed(0)
    core = MultiUniverse(n, K=K, d=192, ctx=CAMPO,
                         enc_layers=4, m=16, nhead=4,
                         tempo=64)
    hist, testa = allena(core, finestre(train, vocab, dev),
                         vocab, dev, passi=PASSI)
    print('train: %.3f -> %.3f' % (hist[0],
                                   sum(hist[-10:]) / 10),
          flush=True)

    W = finestre(test, vocab, dev)
    maschere, ripetizioni = maschere_riuso(test)
    n_r = sum(sum(m[0]) for m in maschere)
    n_d = sum(sum(m[1]) for m in maschere)
    print('riuso: %d posizioni, %d deterministiche '
          '(%.0f%%)' % (n_r, n_d, 100 * n_d / max(n_r, 1)),
          flush=True)
    ripetizioni.sort()
    med = ripetizioni[len(ripetizioni) // 2] \
        if ripetizioni else 0
    over4 = sum(1 for x in ripetizioni if x >= 4)
    print('contesti ripetuti: %d, mediana ripetizioni %d, '
          'con >=4 ripetizioni %d' % (
              len(ripetizioni), med, over4), flush=True)

    z, mem, stati = None, None, []
    for wi, (ids, pad) in enumerate(W):
        X, Y = ids[:, :-1].to(dev), ids[:, 1:].to(dev)
        stati.append((z, mem))
        with torch.no_grad():
            z, h = core(X, pad[:, :-1].to(dev), causal=True,
                        z0=z, mem0=mem)
            mem = core.annota(
                h, sorpresa_simbolo(testa(h), Y),
                pad[:, :-1].to(dev))
    for det in (False, True):
        print('--- %s ---' % (
            'riuso DETERMINISTICO' if det else
            'riuso totale'), flush=True)
        for nome in ('intatto', 'svuotato', 'altro'):
            ce, n = ce_riuso(core, testa, W, maschere, stati,
                             dev, nome, det)
            print('lettore %-8s CE %.3f  (n=%d)' % (
                nome, ce, n), flush=True)
