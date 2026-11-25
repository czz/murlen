"""Scrittura per distinzione: traccia se la posizione
e' lontana dalle tracce gia' presenti.

Il fence (outlier di sorpresa nella finestra) affama
sotto novita' uniforme: ~2-3 tracce/finestra, meta'
dei fatti mai coperti. Alternativa: la novita' si
misura CONTRO IL BUFFER — si percorrono le posizioni
in ordine e si scrive h[p] se la sua distanza dalla
traccia piu' vicina (buffer + gia' scritte in questa
finestra) supera la soglia. Due soglie candidate,
entrambe dai dati:

  'gap':   largest-gap delle distanze minime della
           finestra — divide 'ridondante' da 'nuovo'
  'scala': mediana delle distanze reciproche fra
           tracce del buffer — 'nuovo' = distinto
           quanto le tracce lo sono fra loro

python murlen/tests/check_distinzione.py [cuda|cpu]
"""
import os
import sys

import torch
import torch.nn.functional as Fn

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
    nm = 2 * nfatti
    if g < nm:
        return g // 2
    return ((g - nm) // 2) % nfatti


def _gap(v):
    s = v.sort().values
    if s.numel() < 2 or (s[-1] - s[0]) <= 0:
        return None
    d = s[1:] - s[:-1]
    i = d.argmax().item()
    return float(s[i] + s[i + 1]) / 2


def sel_distinzione(h_b, mem_b, pad_b, modo):
    """riga b: posizioni lontane dalle tracce (buffer
    + scritte in questa finestra), in ordine."""
    hn = Fn.normalize(h_b, dim=-1)
    rif = [] if mem_b is None or mem_b.size(0) == 0 \
        else list(Fn.normalize(mem_b, dim=-1))
    pos = torch.nonzero(~pad_b).flatten().tolist()
    if modo == 'gap' and rif:
        base = torch.stack(rif)
        d = 1 - (hn[pos] @ base.T).max(1).values
        t = _gap(d)
    elif modo == 'scala' and len(rif) > 1:
        base = torch.stack(rif)
        sim = base @ base.T
        sim.fill_diagonal_(float('-inf'))
        t = float((1 - sim.max(1).values).median())
    else:
        t = None
    sel = torch.zeros(h_b.size(0), dtype=torch.bool,
                      device=h_b.device)
    for p in pos:
        if rif:
            d = 1 - float(max(
                (hn[p] * r).sum() for r in rif))
            if t is not None and d <= t:
                continue
        sel[p] = True
        rif.append(hn[p])
    return sel


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
    nomi = ('fence', 'gap', 'scala')
    z, mem = None, None
    cop = {m: [set() for _ in range(seqs.shape[0])]
           for m in nomi}
    scr = {m: [0] * seqs.shape[0] for m in nomi}
    with torch.no_grad():
        for wi, (ids, pad) in enumerate(W):
            X, Y = ids[:, :-1], ids[:, 1:]
            z, h = core(X, pad[:, :-1], causal=True,
                        z0=z, mem0=mem)
            lo = testa(h)
            s = sorpresa_simbolo(lo, Y)
            sels = {
                'fence': core.foglio.scegli_valori(
                    s, pad[:, :-1])}
            for modo in ('gap', 'scala'):
                sels[modo] = torch.stack([
                    sel_distinzione(
                        h[b],
                        mem[b] if mem else None,
                        pad[b, :-1], modo)
                    for b in range(h.size(0))])
            mem = core.annota(h, s, pad[:, :-1])
            for b in range(seqs.shape[0]):
                for modo in nomi:
                    for p in sels[modo][b].nonzero(
                            ).flatten().tolist():
                        f = fatto_di(
                            wi * WIN + p, NFATTI)
                        cop[modo][b].add(f)
                        scr[modo][b] += 1
            print('finestra %2d | scritture '
                  'fence=%s gap=%s scala=%s | '
                  'copertura /32 fence=%s '
                  'gap=%s scala=%s' % (
                      wi,
                      [scr['fence'][b] for b in range(4)],
                      [scr['gap'][b] for b in range(4)],
                      [scr['scala'][b] for b in range(4)],
                      [len(c) for c in cop['fence']],
                      [len(c) for c in cop['gap']],
                      [len(c) for c in cop['scala']]),
                  flush=True)
