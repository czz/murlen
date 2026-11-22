"""Il lettore nasce su testo vero?

Sui compiti-esame il canale foglietto->predizione non si
forma mai (sonda_lettore*: ablazione piatta a ogni numero
di fatti, fence o gate, diretto o curriculum). L'ipotesi
rimasta: il lettore nasce solo con l'uso IN-LINEA — il
fatto riusato dentro il flusso, non a un interrogatorio.

Il testo vero e' il carico in-linea per definizione: qui
lo misuro. Pezzi di libro scorsi a finestre: a ogni
finestra dopo la prima il foglietto porta le tracce delle
precedenti. Le posizioni dove il contesto recente (G
caratteri) e' gia' apparso nel pezzo sono quelle dove la
memoria PUO' servire — criterio meccanico, niente
conoscenza. Tre condizioni sullo stesso z:

- intatto:    le tracce scritte scorrendo QUESTO pezzo
- svuotato:   nessuna traccia (mem0 vuoto)
- episodio-1: le tracce scritte scorrendo un ALTRO pezzo

Se intatto < svuotato alle posizioni di riuso, il lettore
esiste su testo vero; se sono uguali, il foglietto resta
write-only anche sul carico naturale.

python murlen/tests/sonda_lettore_libro.py [cuda|cpu]"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import (vocab_da_sequenze, tensorizza,  # noqa: E402
                          PAD_ID)
from universi_oltre_finestra import (allena,  # noqa: E402
                                     sorpresa_simbolo)
from sonda_campo_libro import (pezzi_liberi,  # noqa: E402
                               CAMPO)
from universi_libro import libri  # noqa: E402
import random

PASSI, K, G = 200, 2, 16


def finestre(seqs, vocab, dev):
    out = []
    for a in range(0, len(seqs[0]), CAMPO):
        ids, pad = tensorizza(
            [s[a:a + CAMPO] for s in seqs], vocab, CAMPO)
        out.append((ids.to(dev), pad.to(dev)))
    return out


def maschere_riuso(seqs):
    """Per ogni posizione globale q: il contesto che finisce
    in q-1 (G simboli) e' apparso prima nel pezzo?"""
    out = []
    for s in seqs:
        visti, m = set(), [False] * len(s)
        for q in range(G, len(s)):
            m[q] = tuple(s[q - G:q]) in visti
            visti.add(tuple(s[q - G:q]))
        out.append(m)
    return out


@torch.no_grad()
def ce_riuso(core, testa, W, maschere, stati, dev, mem_alt):
    """CE media alle posizioni di riuso per finestra>=1,
    con mem0 scelto: 'intatto'|'svuotato'|'altro'."""
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
                if q < len(maschere[b]) and maschere[b][q]:
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
    maschere = maschere_riuso(test)
    n_riuso = sum(sum(m) for m in maschere)
    print('posizioni di riuso: %d' % n_riuso, flush=True)

    z, mem = None, None
    stati = []
    for wi, (ids, pad) in enumerate(W):
        X, Y = ids[:, :-1].to(dev), ids[:, 1:].to(dev)
        stati.append((z, mem))
        with torch.no_grad():
            z, h = core(X, pad[:, :-1].to(dev), causal=True,
                        z0=z, mem0=mem)
            mem = core.annota(
                h, sorpresa_simbolo(testa(h), Y),
                pad[:, :-1].to(dev))
    for nome in ('intatto', 'svuotato', 'altro'):
        ce, n = ce_riuso(core, testa, W, maschere, stati,
                         dev, nome)
        print('lettore %-8s CE riuso %.3f  (n=%d)' % (
            nome, ce, n), flush=True)
