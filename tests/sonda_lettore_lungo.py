"""Il lettore su libro LUNGO, allenato a sufficienza.

sonda_scala_indirizzo: il recall indirizzato ESISTE —
a 2000 passi sul flusso denso sintetico il foglietto
viene consultato per contenuto (prima 0.41 vs 3.82/6.81).
Domanda aperta: il riuso naturale dei libri offre
abbastanza letture indirizzate per accenderlo?

Stesso protocollo di sonda_rif_lontani ma sulle
condizioni dove il lettore e' nato: testo di romanzi
lunghi (~/ncm-lab/libri_lunghi), pezzi da 8 finestre,
d=384/8layers, 2000 passi. Le posizioni misurate sono
il riuso oltre CAMPO deterministico — dove il contesto
non basta e la traccia e' l'unica via.

Ablazioni: intatto / svuotato / episodio-1 / senza-z.

python murlen/tests/sonda_lettore_lungo.py [cuda|cpu] [passi]
"""
import glob
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import PAD_ID  # noqa: E402
from murlen.ponte import vocab_da_sequenze, tensorizza  # noqa: E402
from universi_oltre_finestra import (allena,  # noqa: E402
                                     sorpresa_simbolo)
from sonda_rif_lontani import maschere_lontane  # noqa: E402

LIBRI_LUNGHI = os.path.expanduser(
    '~/ncm-lab/libri_lunghi')
CAMPO, K, G = 128, 2, 16
L = 8 * CAMPO     # 1024 caratteri, 7 finestre valutabili


def libri_lunghi():
    out = []
    for f in sorted(glob.glob(
            os.path.join(LIBRI_LUNGHI, '*.txt'))):
        out.append(open(f, encoding='utf-8',
                        errors='replace').read())
    return out


def pezzi(testi, ntrain, ntest, lun=L):
    """Pezzi di lun caratteri da tutti i libri; ogni 5o
    va al test — il testo di prova non e' mai stato
    visto e copre tutti i libri (la densita' di riuso
    varia molto da libro a libro)."""
    out = []
    for t in testi:
        for a in range(0, len(t) - lun, lun):
            out.append([ord(c) for c in t[a:a + lun]])
    test = out[::5][:ntest]
    train = [s for i, s in enumerate(out) if i % 5]
    return train, test


def finestre(seqs, vocab, dev):
    out = []
    for a in range(0, len(seqs[0]), CAMPO):
        ids, pad = tensorizza(
            [s[a:a + CAMPO] for s in seqs], vocab, CAMPO)
        out.append((ids.to(dev), pad.to(dev)))
    return out


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
        elif mem_alt == 'senza-z':
            mem0 = mem_in
            z_in = None
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
    passi = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    testi = libri_lunghi()
    print('%d libri lunghi' % len(testi), flush=True)
    train, test = pezzi(testi, 0, 128)
    print('%d pezzi train, %d test (L=%d)' % (
        len(train), len(test), L), flush=True)
    vocab, n = vocab_da_sequenze(train + test)
    torch.manual_seed(0)
    core = MultiUniverse(n, K=K, d=384, ctx=CAMPO,
                         enc_layers=8, m=32, nhead=8,
                         tempo=64)
    hist, testa = allena(core, finestre(train, vocab, dev),
                         vocab, dev, passi=passi)
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
    for det in (True, False):
        print('--- riuso lontano %s ---' % (
            'deterministico' if det else 'totale'),
            flush=True)
        for nome in ('intatto', 'svuotato', 'altro',
                     'senza-z'):
            ce, n = ce_pos(core, testa, W, maschere, stati,
                           dev, nome, det)
            print('lettore %-8s CE %.3f  (n=%d)' % (
                nome, ce, n), flush=True)
