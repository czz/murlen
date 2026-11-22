"""Prova di memoria a lungo raggio: la chiave.
All'inizio della sequenza ci sono NBIT simboli-chiave, poi
parole qualsiasi con NZONE marcatori: dopo il marcatore i
il simbolo giusto dipende dal bit i%NBIT della chiave.
Nella seconda finestra la chiave non si vede e ogni bit
serve una volta sola: o z l'ha portato o si indovina.
Misura la loss della seconda finestra in tutto e sui soli
simboli dopo i marcatori, con z portato, dimenticato,
mescolato fra sequenze; e zittendo comm e universi.
python murlen/tests/universi_chiave.py [cuda|cpu]"""
import copy
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import vocab_da_sequenze  # noqa: E402
from universi_oltre_finestra import (  # noqa: E402
    PASSI, K, allena, _ce)
from murlen.ponte import tensorizza  # noqa: E402


NBIT, NZONE = 16, 32        # bit; marche 0-7 tre volte
                            # in finestra 1, 8-15 una
                            # volta in finestra 2
WIN = 256                   # finestre corte: il trasporto
                            # di z non dipende dalla
                            # dimensione, e cosi' il batch
                            # torna grande (piu' dati)


def finestre(seqs, vocab, dev, win=WIN):
    """Ogni sequenza -> finestre di win simboli."""
    L = len(seqs[0])
    out = []
    for a in range(0, L, win):
        ids, pad = tensorizza([s[a:a + win] for s in seqs],
                              vocab, win)
        out.append((ids.to(dev), pad.to(dev)))
    return out
KEY, MARK = 100, 300        # bit: 60+b; marca: 300+i
WA, WB = [42, 43, 44, 45], [46, 47, 48, 49]  # risposte
RIEMP = [[2, 3], [4, 5], [6, 7], [8, 9]]     # riempitivo:
                            # ciclo fisso, costa ~0 —
                            # tutto l'errore viene dalle
                            # risposte dopo le marche


def dati_chiave(n, seed, L=2 * WIN):
    """Chiave per contenuto all'inizio: coppie [marca i,
    bit i]. Le marche 0-7, tre volte ciascuna, stanno nella
    prima finestra (chiave visibile: si impara il
    meccanismo); le marche 8-15 nella seconda — ogni bit
    serve UNA volta sola e non si rivede: la fonte
    possibile e' solo z. Dopo la marca i la risposta e'
    WA se bit i e' 0, altrimenti WB."""
    r = random.Random(seed)
    out = []
    for _ in range(n):
        bits = [r.randrange(2) for _ in range(NBIT)]
        s = []
        for i in range(NBIT):
            s += [MARK + i, KEY + bits[i]]
        j = 0
        nw1 = NZONE - NBIT // 2        # 24 zone in fin. 1
        for i in range(NZONE):
            if i < nw1:                        # finestra 1:
                m = i % (NBIT // 2)            # marche 0-7
                fine = 2 * NBIT + (i + 1) * (  # ripetute 3x
                    (WIN - 2 * NBIT) // nw1) - 4
            else:                              # finestra 2:
                m = NBIT // 2 + i - nw1        # marche 8-15
                fine = WIN + (i - nw1 + 1) * (  # una volta
                    (L - WIN) // (NBIT // 2)) - 8
            while len(s) < fine:
                s += RIEMP[j % 4] + [1]
                j += 1
            s += [MARK + m] + (WA if not bits[m] else WB)
        while len(s) < L:
            s += RIEMP[j % 4] + [1]
            j += 1
        out.append(s[:L])
    return out


@torch.no_grad()
def valuta(core, testa, W, dev, vocab):
    """Loss finestra per finestra; sulla seconda anche la
    loss sui soli simboli dopo i marcatori (dove serve la
    chiave)."""
    core.eval()
    ids1, p1 = W[0]
    ids2, p2 = W[1]
    X1, Y1 = ids1[:, :-1], ids1[:, 1:]
    X2, Y2 = ids2[:, :-1], ids2[:, 1:]
    # posizioni di X2 con una marca: li' Y e' il primo
    # simbolo della risposta, l'unico imprevedibile
    marche = torch.tensor([vocab[MARK + i] for i in
                           range(NBIT)], device=dev)
    scelta = torch.isin(X2, marche)
    scelta1 = torch.isin(X1, marche)

    # prima marca della finestra: nessun indizio locale,
    # la risposta puo' venire solo da z
    prima = torch.zeros_like(scelta)
    prima[:, scelta.float().argmax(1)] = True

    def seconda(m, z0):
        _, h2 = m(X2, p2[:, :-1], causal=True, z0=z0)
        lo = testa(h2)
        lc = nn.functional.cross_entropy(
            lo[scelta], Y2[scelta]).item()
        lp = nn.functional.cross_entropy(
            lo[prima], Y2[prima]).item()
        return _ce(lo, Y2).item(), lc, lp

    def prova(nome, f=None, modo='porta'):
        m = copy.deepcopy(core).to(dev)
        if f:
            with torch.no_grad():
                f(m)
        m.eval()
        z1 = m(X1, p1[:, :-1], causal=True)[0]
        if modo == 'dimentica':
            z1 = None
        elif modo == 'mescola':
            z1 = z1.roll(1, 0)
        l2, lc, lp = seconda(m, z1)
        print('%-42s %5.3f  %5.3f  %5.3f' % (nome, l2, lc, lp),
              flush=True)
        del m
        if dev == 'cuda':
            torch.cuda.empty_cache()

    _, h1 = core(X1, p1[:, :-1], causal=True)
    lo1 = testa(h1)
    print('finestra 1 (riferimento): loss %.3f  dopo marca %.3f'
          % (_ce(lo1, Y1).item(),
             nn.functional.cross_entropy(
                 lo1[scelta1], Y1[scelta1]).item()))
    z, _ = core(X1, p1[:, :-1], causal=True)
    v = nn.functional.normalize(z.flatten(2), dim=-1)
    sim = (v @ v.transpose(1, 2))[:, *torch.triu_indices(
        K, K, 1)].mean().item()
    print('somiglianza fra universi: %.3f' % sim)
    print('marche in finestra 2: %d a sequenza'
          % (scelta.sum(1).float().mean().item()))
    print('%-42s %5s  %9s  %s' % ('finestra 2', 'loss',
                                   'dopo marca', '1a marca'))
    prova('z portato dalla finestra 1')
    prova('z dimenticato (z0=None)', modo='dimentica')
    prova('z mescolato fra le sequenze', modo='mescola')
    prova('senza comunicazione (lam=0)',
          lambda m: m.comm.lam.zero_())
    prova('universi muti (zinj=0)',
          lambda m: m.zinj.weight.zero_())
    prova('lettura muta (zread=0)',
          lambda m: m.zread.out_proj.weight.zero_())
    d = core.zinj.out_features
    for k in range(core.K):
        prova('universo %d muto' % k,
              lambda m, k=k: m.zinj.weight[
                  :, k * d:(k + 1) * d].zero_())


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = int(sys.argv[2]) if len(sys.argv) > 2 else PASSI
    train = dati_chiave(passi * 8, 1)
    test = dati_chiave(64, 2)
    vocab, n = vocab_da_sequenze(train + test)
    torch.manual_seed(0)
    core = MultiUniverse(n, K=K)
    hist, testa = allena(core, finestre(train, vocab, dev),
                         vocab, dev, passi=passi, batch=8)
    torch.save({'core': core.state_dict(),
                'testa': testa.state_dict(),
                'vocab': vocab},
               os.path.join(os.path.dirname(
                   os.path.abspath(__file__)),
                   'dati', 'universi_chiave.pt'))
    print('train: %.3f -> %.3f' % (hist[0],
                                   sum(hist[-10:]) / 10))
    valuta(core, testa, finestre(test, vocab, dev), dev, vocab)
