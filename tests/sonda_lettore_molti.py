"""Il lettore generalizza su molti fatti?

sonda_densita_libro: un romanzo vero offre ~157k letture
lontane deterministiche — ma sono ~1 per contesto. Il
ginocchio misurato (~4) era lo STESSO fatto letto 4 volte.
Resta la domanda: il canale di lettura e' un circuito
condiviso — si allena anche con N fatti diversi letti
poche volte ciascuno, o la densita' serve per singolo
fatto?

N fatti in testa [MARCA+i, VAL+v_i], poi il flusso:
giunzioni [MARCA+q, GIUN+v_q] distribuite dopo la testa —
le prime in finestra 1 (fatto in contesto: la lezione),
le altre in finestre dopo (la traccia e' l'unica via: la
misura). usi letture per fatto; N in {1, 8, 32}.
Ablazione intatto/svuotato/episodio-1 sulle giunzioni
delle finestre >= 1.

Se il lettore nasce a N=32 come a N=1 -> il circuito
generalizza: 63+ eventi/libro bastano, "leggere tanto"
funziona. Se muore gia' a N=8 -> la densita' serve per
fatto: il carico naturale non basta.

python murlen/tests/sonda_lettore_molti.py [cuda|cpu] [passi]
"""
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import murlen.model as MM  # noqa: E402
from murlen.ponte import (vocab_da_sequenze, tensorizza,  # noqa: E402
                          PAD_ID)
from universi_oltre_finestra import (  # noqa: E402
    K, allena, sorpresa_simbolo)

WIN = 256
NFATTI_MAX, NVAL = 32, 20
MARK, ANS, GIUN = 300, 400, 450
CNTA, CNTB = 10, 20
L = 4 * WIN


def dati(n, seed, nfatti, usi):
    """Episodio: testa di nfatti coppie [MARK+i, ANS+v_i];
    poi coppie contatori con giunzioni [MARK+q, GIUN+v_q]
    distanziate 0-2 coppie — ogni fatto letto usi volte,
    ordine mescolato."""
    r = random.Random(seed)
    out = []
    for _ in range(n):
        vals = [r.randrange(NVAL) for _ in range(nfatti)]
        s = []
        for i in range(nfatti):
            s += [MARK + i, ANS + vals[i]]
        query = [i for i in range(nfatti)
                 for _ in range(usi)]
        r.shuffle(query)
        # giunzioni a intervalli uniformi su TUTTA la
        # sequenza: un quarto circa in finestra 1 (fatto in
        # contesto), il resto oltre — letture da memoria
        inizio = 2 * nfatti + 8
        spazi = (L - inizio) // max(1, len(query))
        ca, cb = r.randrange(8), r.randrange(8)
        qi = 0
        while len(s) < L:
            if qi < len(query) and \
                    len(s) >= inizio + spazi * qi:
                q = query[qi]
                s += [MARK + q, GIUN + vals[q]]
                qi += 1
            else:
                s += [CNTA + ca % 8, CNTB + cb % 8]
                ca += 1
                cb += 1
        out.append(s[:L])
    return out


def finestre(seqs, vocab, dev):
    out = []
    for a in range(0, L, WIN):
        ids, pad = tensorizza(
            [s[a:a + WIN] for s in seqs], vocab, WIN)
        out.append((ids.to(dev), pad.to(dev)))
    return out


@torch.no_grad()
def lettore(core, testa, test, vocab, dev):
    """CE alle risposte di giunzione (GIUN+..) nelle
    finestre >= 1, mem0 per condizione."""
    core.eval()
    giun = torch.tensor([vocab[GIUN + i]
                         for i in range(NVAL)], device=dev)
    W = finestre(test, vocab, dev)
    z, mem, stati = None, None, []
    for ids, pad in W:
        X, Y = ids[:, :-1].to(dev), ids[:, 1:].to(dev)
        stati.append((z, mem))
        z, h = core(X, pad[:, :-1].to(dev), causal=True,
                    z0=z, mem0=mem)
        mem = core.annota(h, sorpresa_simbolo(testa(h), Y),
                          pad[:, :-1].to(dev))
    for nome in ('intatto', 'svuotato', 'episodio-1'):
        tot, n = 0., 0
        for wi, (ids, pad) in enumerate(W):
            if wi == 0:
                continue
            X, Y = ids[:, :-1].to(dev), ids[:, 1:].to(dev)
            z_in, mem_in = stati[wi]
            if nome == 'svuotato':
                mem0 = None
            elif nome == 'episodio-1':
                mem0 = [mem_in[(b + 1) % len(mem_in)]
                        for b in range(len(mem_in))]
            else:
                mem0 = mem_in
            _, h = core(X, pad[:, :-1].to(dev), causal=True,
                        z0=z_in, mem0=mem0)
            lo = testa(h)
            sel = torch.isin(Y, giun)
            if sel.any():
                tot += nn.functional.cross_entropy(
                    lo[sel], Y[sel]).item() * sel.sum().item()
                n += sel.sum().item()
        caso = torch.log(torch.tensor(float(NVAL))).item()
        print('lettore %-10s giunzioni %.3f  (n=%d, a caso'
              ' %.3f)' % (nome, tot / max(n, 1), n, caso),
              flush=True)


COND = [
    # nome, nfatti, usi
    ('N=1  usi=8', 1, 8),
    ('N=8  usi=8', 8, 8),
    ('N=32 usi=8', 32, 8),
]

if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    for nome, nf, usi in COND:
        print('=== %s ===' % nome, flush=True)
        test = dati(64, 2, nf, usi)
        t0 = dati(8, 9, nf, usi)
        # copertura: il vocab deve contenere OGNI simbolo
        # possibile, non solo quelli usciti nel campione
        tutti = [list(range(0, GIUN + NVAL + 2))]
        vocab, n = vocab_da_sequenze(t0 + test + tutti)
        torch.manual_seed(0)
        core = MM.MultiUniverse(n, K=K)
        train = dati(passi * 8, 1, nf, usi)
        hist, testa = allena(
            core, finestre(train, vocab, dev), vocab, dev,
            passi=passi, batch=8)
        print('train: %.3f -> %.3f' % (
            hist[0], sum(hist[-10:]) / 10), flush=True)
        lettore(core, testa, test, vocab, dev)
        del core, testa
        if dev == 'cuda':
            torch.cuda.empty_cache()
