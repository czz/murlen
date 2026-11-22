"""scala — le capacita' congenite, misurate.

d, strati, ctx, m, nhead, T, tempo, MEM_CAP sono
stati DICHIARATI, mai misurati. Qui un asse alla
volta scala sullo stesso compito — fatti da
riusare a distanza su dati multi-modo — e si
guarda dove la curva si ferma: la taglia giusta
e' il ginocchio, non il massimo.

Compito: sequenze a 3 argomenti latenti (multi-
modo, per K e comm). Dentro: FATTI (chiave ->
richiamo dopo DIST simboli) intervallati da
filler dell'argomento. Il richiamo serve a
mem/z quando la chiave e' uscita dalla
finestra; il numero di fatti aperti stressa
MEM_CAP. Misura: CE alle posizioni di richiamo
(il trasporto) e CE totale.

Uso: python tests/scala.py [asse ...]
Default: tutti gli assi, 2 seed.
"""
import os
import random
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen import model as _mod            # noqa: E402
from murlen.model import MultiUniverse      # noqa: E402
from murlen.ponte import (vocab_da_sequenze,  # noqa: E402
                          tensorizza,
                          TestaPredizione, PAD_ID)

TOPICI, FATTI, DIST = 3, 200, 200
L, N_TRAIN, N_EVAL = 1024, 8, 4
PASSI, BATCH = 120, 1
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

DEF = dict(d=384, enc_layers=8, ctx=1024, m=32,
           nhead=8, T=4, tempo=64, MEM_CAP=256,
           jitter=0.1, K=2, lr=3e-4, clip=1.)
ASSI = {
    'd': [96, 384, 768],
    'enc_layers': [4, 8, 12],
    'ctx': [256, 1024, 2048],
    'm': [8, 32, 96],
    'nhead': [2, 8, 16],
    'T': [2, 4, 8],
    'tempo': [32, 64, 256],
    'MEM_CAP': [64, 256, 1024],
    'jitter': [0., .1, .3],
    'K': [1, 2, 8],
    'lr': [1e-4, 3e-4, 1e-3],
    'clip': [.5, 1., 4.],
}


def dati(n, seed):
    """Sequenze di L simboli a 3 argomenti. In
    mezzo: [chiave] + DIST-1 filler d'argomento +
    [chiave] — il secondo avvistamento e' la
    posizione di richiamo (indici restituiti)."""
    r = random.Random(seed)
    out, richiami = [], []
    for _ in range(n):
        s, pos = [], []
        while len(s) < L - DIST - 2:
            arg = r.randrange(TOPICI)
            key = r.randrange(FATTI)
            s += [2 + arg, 10 + key]
            for _ in range(DIST - 1):
                s.append(50 + arg * 200 +
                         r.randrange(200))
            pos.append(len(s))
            s.append(10 + key)
        out.append(s[:L])
        richiami.append(pos)
    return out, richiami


def finestre(seqs, vocab, ctx):
    L = len(seqs[0])
    out = []
    for a in range(0, L, ctx):
        ids, pad = tensorizza(
            [s[a:a + ctx] for s in seqs], vocab, ctx)
        out.append((ids.to(DEV), pad.to(DEV)))
    return out


def _sorpresa(lo, Y):
    B, N = Y.shape
    s = torch.full((B, 1), float('inf'),
                   device=lo.device)
    ce = nn.functional.cross_entropy(
        lo[:, :-1].flatten(0, 1), Y[:, :-1].flatten(),
        reduction='none', ignore_index=PAD_ID
        ).view(B, -1)
    return torch.cat([s, ce], 1)


def passo(core, testa, opt, W, sel, clip):
    """Finestre in sequenza: z e foglietto
    attraversano i bordi."""
    z, mem, loss = None, None, 0.
    for ids, pad in W:
        X, Y = ids[sel, :-1], ids[sel, 1:]
        z, h = core(X, pad[sel, :-1], causal=True,
                    z0=z, mem0=mem)
        lo = testa(h)
        loss = loss + nn.functional.cross_entropy(
            lo.flatten(0, 1), Y.flatten(),
            ignore_index=PAD_ID)
        mem = core.annota(h, _sorpresa(lo, Y),
                          pad[sel, :-1])
    opt.zero_grad()
    loss.backward()
    ng = nn.utils.clip_grad_norm_(
        list(core.parameters()) +
        list(testa.parameters()), clip)
    if torch.isfinite(ng):
        opt.step()
    else:
        opt.zero_grad()
    return loss.item()


def valuta(core, testa, seqs, richiami, vocab, ctx):
    """CE totale + CE alle posizioni di
    richiamo (chiave ri-vista dopo DIST)."""
    core.eval()
    with torch.no_grad():
        W = finestre(seqs, vocab, ctx)
        z, mem, tot, n = None, None, 0., 0
        lo_tot = None
        # posizioni di richiamo globali ->
        # (finestra, offset)
        for wi, (ids, pad) in enumerate(W):
            X, Y = ids[:, :-1], ids[:, 1:]
            z, h = core(X, pad[:, :-1], causal=True,
                        z0=z, mem0=mem)
            lo = testa(h)
            tot += nn.functional.cross_entropy(
                lo.flatten(0, 1), Y.flatten(),
                ignore_index=PAD_ID,
                reduction='sum').item()
            n += (Y != PAD_ID).sum().item()
            lo_tot = (lo_tot or []) + [
                (wi, lo, Y)]
        # CE ai richiami: posizione p nella
        # sequenza -> finestra p//ctx, col p%ctx
        ric, nr = 0., 0
        for b, pos in enumerate(richiami):
            for p in pos:
                wi, off = divmod(p, ctx)
                lo, Y = lo_tot[wi][1], lo_tot[wi][2]
                # il simbolo in posizione off si
                # predice da lo[off-1] vs Y[off-1]
                if off < 1 or off - 1 >= Y.size(1):
                    continue
                if Y[b, off - 1] == PAD_ID:
                    continue
                ric += nn.functional.cross_entropy(
                    lo[b, off - 1:off],
                    Y[b, off - 1:off],
                    ignore_index=PAD_ID).item()
                nr += 1
    core.train()
    return tot / n, ric / max(nr, 1)


def prova(asse, valore, seed):
    cfg = dict(DEF)
    cfg[asse] = valore
    _mod.MEM_CAP = cfg['MEM_CAP']
    torch.manual_seed(seed)
    seqs, richiami = dati(N_TRAIN, seed)
    ev, ev_rich = dati(N_EVAL, seed + 999)
    vocab, nv = vocab_da_sequenze(seqs + ev)
    core = MultiUniverse(
        nv, K=cfg['K'], T=cfg['T'], init='projected',
        enc_layers=cfg['enc_layers'],
        d=cfg['d'], m=cfg['m'], nhead=cfg['nhead'],
        ctx=cfg['ctx'], tempo=cfg['tempo'],
        jitter=cfg['jitter']).to(DEV)
    W = finestre(seqs, vocab, cfg['ctx'])
    testa = TestaPredizione(
        nv, cfg['d']).to(DEV)
    opt = torch.optim.AdamW(
        list(core.parameters()) +
        list(testa.parameters()), lr=cfg['lr'])
    core.train()
    for _ in range(PASSI):
        sel = torch.randperm(N_TRAIN)[:BATCH]
        passo(core, testa, opt, W, sel, cfg['clip'])
    ce, ce_ric = valuta(core, testa, ev, ev_rich,
                        vocab, cfg['ctx'])
    del core, testa, opt, W
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    return ce, ce_ric


def main():
    assi = sys.argv[1:] or list(ASSI)
    print('scala — capacita\' congenite misurate')
    print('dev=%s  compito: %d fatti a dist. %d, '
          'L=%d, %d argomenti, passi=%d' % (
              DEV, FATTI, DIST, L, TOPICI, PASSI))
    print()
    print('%-12s %-8s %8s %8s %8s' % (
        'asse', 'valore', 'ce', 'ce_ric', 'seed'))
    for asse in assi:
        for v in ASSI[asse]:
            ces, rics = [], []
            t0 = time.time()
            for seed in (0, 1):
                try:
                    ce, ric = prova(asse, v, seed)
                except torch.cuda.OutOfMemoryError:
                    # il soffitto hardware e'
                    # una misura: questa taglia
                    # non ci sta su questa macchina
                    ce, ric = float('nan'), \
                        float('nan')
                ces.append(ce)
                rics.append(ric)
            oom = any(c != c for c in ces)
            if oom:
                print('%-12s %-8s %8s %8s %8s' % (
                    asse, v, 'OOM', 'OOM', '0,1'),
                    flush=True)
            else:
                print('%-12s %-8s %8.3f %8.3f %8s' % (
                    asse, v, sum(ces) / len(ces),
                    sum(rics) / len(rics),
                    '0,1'), flush=True)
            _ = t0
    _mod.MEM_CAP = DEF['MEM_CAP']


if __name__ == '__main__':
    main()
