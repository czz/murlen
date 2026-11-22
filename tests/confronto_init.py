"""Confronto init='learned' contro init='projected'.
Stessi dati, stessi semi; misura loss su sequenze mai viste,
diversita' fra universi, peso degli universi nella previsione,
velocita' e memoria.
python murlen/tests/confronto_init.py [cuda|cpu] [lr] [--semi 0,1,2] [--cresci]
--cresci: si parte da 1 universo e si arriva a K con cresci(),
un terzo dei passi per ogni fase.  (dalla cartella sopra murlen)"""
import os
import random
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse, param_count  # noqa: E402
from murlen.ponte import (vocab_da_sequenze, impara_predizione,  # noqa: E402
                          tensorizza, _flusso, PAD_ID)

SEMI, PASSI, L, K = (0, 1, 2), 200, 256, 3
LR = 3e-4


def dati(n, seed):
    """Frasi di 'parole' (3-6 simboli); un argomento scelto
    all'inizio decide quali parole compaiono nella sequenza."""
    r = random.Random(0)
    parole = [[r.randrange(2, 42) for _ in range(r.randint(3, 6))]
              for _ in range(60)]
    r = random.Random(seed)
    out = []
    for _ in range(n):
        arg = r.randrange(3)
        s = [100 + arg]
        while len(s) < L:
            s += parole[arg * 20 + r.randrange(20)] + [1]
        out.append(s[:L])
    return out


@torch.no_grad()
def valuta(core, testa, seqs, vocab, dev):
    core.eval()
    ids, pad = tensorizza(seqs, vocab, L)
    ids, pad = ids.to(dev), pad.to(dev)
    X, Y = ids[:, :-1], ids[:, 1:]
    loss = nn.functional.cross_entropy(
        testa(_flusso(core, X, pad[:, :-1])).flatten(0, 1),
        Y.flatten(), ignore_index=PAD_ID).item()
    z = core(X, pad[:, :-1], causal=True)[0]           # B,K,M,d
    v = nn.functional.normalize(z.flatten(2), dim=-1)  # B,K,M*d
    sim = v @ v.transpose(1, 2)
    i, j = torch.triu_indices(K, K, 1)
    return loss, sim[:, i, j].mean().item()


def prova(init, seed, train, test, vocab, n, dev, cresci=False):
    torch.manual_seed(seed)
    fasi = K if cresci else 1
    core = MultiUniverse(n, K=1 if cresci else K, init=init)
    if dev == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    hist, testa = [], None
    for f in range(fasi):
        if f:
            core.cresci()
        h, testa = impara_predizione(core, train, vocab,
                                     passi=PASSI // fasi, device=dev,
                                     lr=LR, batch=16, seed=seed + f,
                                     testa=testa)
        hist += h
    if dev == 'cuda':
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / len(hist)
    loss, sim = valuta(core, testa, test, vocab, dev)
    vram = (torch.cuda.max_memory_allocated() / 2**30
            if dev == 'cuda' else float('nan'))
    return dict(train=sum(hist[-10:]) / 10, test=loss, sim=sim,
                voce=core.zinj.weight.norm().item(),
                par=param_count(core) / 1e6, s=dt, vram=vram)


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    if len(sys.argv) > 2:
        LR = float(sys.argv[2])
    cresci = '--cresci' in sys.argv
    if '--semi' in sys.argv:
        SEMI = tuple(int(x) for x in
                     sys.argv[sys.argv.index('--semi') + 1].split(','))
    train, test = dati(PASSI * 16, 1), dati(64, 2)
    vocab, n = vocab_da_sequenze(train + test)
    ris = {}
    for init in ('learned', 'projected'):
        for s in SEMI:
            r = prova(init, s, train, test, vocab, n, dev, cresci)
            ris.setdefault(init, []).append(r)
            print(init, s, {k: round(v, 4) for k, v in r.items()},
                  flush=True)
    print('\nmedia su %d semi (K=%d, tempo=64, comm, %d passi, '
          'sequenze da %d, lr=%g%s):' % (len(SEMI), K, PASSI, L, LR,
                                     ', cresci da 1' if cresci else ''))
    print('%-10s %8s %8s %10s %8s %8s %8s %8s' % (
        'init', 'train', 'test', 'somigl.', 'voce', 'Mpar',
        's/passo', 'VRAM GB'))
    for init, rr in ris.items():
        m = {k: sum(r[k] for r in rr) / len(rr) for k in rr[0]}
        print('%-10s %8.3f %8.3f %10.3f %8.3f %8.1f %8.2f %8.2f' % (
            init, m['train'], m['test'], m['sim'], m['voce'], m['par'],
            m['s'], m['vram']))
