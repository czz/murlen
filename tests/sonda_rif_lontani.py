"""Il riuso LONTANO nei libri: dove la memoria e' l'unica via.

sonda_carico_libro: il riuso a 16 caratteri e' 86%
deterministico -> la mappa contesto->continuazione e' una
funzione fissa, imparabile nei pesi: la traccia non serve.
E ogni contesto si ripete in mediana 2 volte — sotto il
ginocchio di ~4 letture misurato per accendere il canale.

La posizione dove la memoria e' DAVVERO l'unica via: il
contesto G e' apparso solo a distanza > CAMPO — la prima
occorrenza e' fuori dalla finestra corrente, invisibile
all'encoder; la coppia contesto->continuazione e' rara nel
pezzo (non imparabile nei pesi). Un nome detto una volta,
un numero, un riferimento: prevederlo richiede la traccia
scritta quando e' passato.

Qui si misura: quante sono per pezzo, a che distanza
mediana, e la CE alle posizioni lontane per
intatto/svuotato/altro — il primo test dove la memoria sul
libro e' potenzialmente necessaria, non decorativa.

python murlen/tests/sonda_rif_lontani.py [cuda|cpu]"""
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import PAD_ID  # noqa: E402
from murlen.ponte import vocab_da_sequenze  # noqa: E402
from universi_oltre_finestra import (allena,  # noqa: E402
                                     sorpresa_simbolo)
from sonda_campo_libro import pezzi_liberi, CAMPO  # noqa: E402
from sonda_lettore_libro import finestre  # noqa: E402
from universi_libro import libri  # noqa: E402

PASSI, K, G = 200, 2, 16


def maschere_lontane(seqs):
    """riuso lontano: il contesto G prima di q e' apparso
    prima, e la sua prima occorrenza e' a distanza > CAMPO
    (fuori finestra). deterministico = stessa
    continuazione della prima volta.
    -> maschere (riu, det), distanze"""
    out, distanze = [], []
    for s in seqs:
        visti = {}
        riu, det = [False] * len(s), [False] * len(s)
        for q in range(G, len(s)):
            c = tuple(s[q - G:q])
            if c in visti:
                sym0, q0 = visti[c]
                if q - q0 > CAMPO:
                    riu[q] = True
                    det[q] = sym0 == s[q]
                    distanze.append(q - q0)
            else:
                visti[c] = (s[q], q)
        out.append((riu, det))
    return out, distanze


@torch.no_grad()
def ce_pos(core, testa, W, maschere, stati, dev,
           mem_alt, det_solo):
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
    maschere, distanze = maschere_lontane(test)
    n_r = sum(sum(m[0]) for m in maschere)
    n_d = sum(sum(m[1]) for m in maschere)
    distanze.sort()
    med = distanze[len(distanze) // 2] if distanze else 0
    per_pezzo = [sum(m[0]) for m in maschere]
    print('riuso LONTANO (>CAMPO): %d posizioni, '
          '%d deterministiche (%.0f%%), mediana distanza '
          '%d, media %.1f per pezzo' % (
              n_r, n_d, 100 * n_d / max(n_r, 1), med,
              n_r / len(test)), flush=True)
    print('pezzi con >=4 posizioni lontane: %d/%d' % (
        sum(1 for x in per_pezzo if x >= 4), len(test)),
        flush=True)

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
        print('--- riuso lontano %s ---' % (
            'deterministico' if det else 'totale'),
            flush=True)
        for nome in ('intatto', 'svuotato', 'altro'):
            ce, n = ce_pos(core, testa, W, maschere, stati,
                           dev, nome, det)
            print('lettore %-8s CE %.3f  (n=%d)' % (
                nome, ce, n), flush=True)
