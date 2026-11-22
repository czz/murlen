"""Prova del dialogo: due flussi intrecciati con stati privati.
In testa alla sequenza i due modi [M_A+a, M_B+b] (a,b in 0..3).
Poi coppie alternate: flusso A = contatore 10+i, flusso B =
contatore 20+j — entrambi banali, nessuno rivela il suo modo.
Ogni JUN_OGNI coppie il flusso A emette una giunzione
400 + a*4 + b: prevederla serve la COMBINAZIONE dei due stati
privati. In finestra 1 i modi sono visibili: si impara il
meccanismo; in finestra 2 sono fuori contesto — la fonte
possibile e' solo z e, se i fatti stanno in universi
diversi, la comm. Le giunzioni dopo la prima in finestra 2
si possono copiare per induzione: la misura pulita e' la
prima. Ablazioni: z dimenticato/mescolato, comm muta,
universi muti.
Senza curriculum i modi in testa non arrivano mai oltre il
confine: con il curriculum (pos_modi 200->0) il meccanismo
della giunzione si impara e la comm diventa load-bearing
(zittirla raddoppia la loss di W2); i fatti dei modi restano
non trasportati (1a giunzione ~caso).
python murlen/tests/universi_dialogo.py [cuda|cpu] [passi]"""
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
    K, allena, _ce, sorpresa_simbolo)
from universi_chiave import finestre  # noqa: E402

WIN = 256                     # come le altre prove: batch grande
NMOD = 4                      # modi per flusso -> 16 giunzioni
MA, MB, GIUN = 300, 310, 400
CNTA, CNTB = 10, 20
JUN_OGNI, JUN_FASE = 8, 5     # una giunzione ogni 8 coppie,
                            # fase 5: nessuna giunzione cade
                            # sul confine (pos 256 starebbe
                            # nell'input di W2 e svuoterebbe
                            # la misura: il codice sarebbe
                            # gia' visibile)


def dati_dialogo(n, seed, L=2 * WIN, pos_modi=0,
                 un_fatto=False):
    """pos_modi: dove stanno i modi — il curriculum li
    allontana dal confine piano piano. un_fatto: un solo
    modo in testa e la giunzione dipende solo da lui —
    isola il trasporto di UN fatto dal binding di due."""
    r = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = r.randrange(NMOD), r.randrange(NMOD)
        s = [30] * pos_modi + ([MA + a] if un_fatto else
                               [MA + a, MB + b])
        ca, cb = r.randrange(8), r.randrange(8)
        p = 0
        while len(s) < L:
            if p % JUN_OGNI == JUN_FASE:
                s.append(GIUN + (a if un_fatto else
                                 a * NMOD + b))   # giunzione nel flusso A
            else:
                s.append(CNTA + ca % 8)
            ca += 1                             # il contatore
            s.append(CNTB + cb % 8)             # avanza comunque
            cb += 1
            p += 1
        out.append(s[:L])
    return out


@torch.no_grad()
def valuta(core, testa, W, dev, vocab, un_fatto=False):
    """Loss alle giunzioni: serve a*4+b, cioe' i due modi
    insieme. In finestra 1 (modi visibili) e' il controllo
    del meccanismo; la prima giunzione di finestra 2 e' la
    misura pulita — dopo, il codice si puo' copiare."""
    core.eval()
    ids1, p1 = W[0]
    ids2, p2 = W[1]
    X1, Y1 = ids1[:, :-1], ids1[:, 1:]
    X2, Y2 = ids2[:, :-1], ids2[:, 1:]
    ngiun = NMOD if un_fatto else NMOD * NMOD
    giun = torch.tensor([vocab[GIUN + i] for i in
                         range(ngiun)], device=dev)
    sel1 = torch.isin(Y1, giun)
    sel2 = torch.isin(Y2, giun)
    prima = torch.zeros_like(sel2)
    prima[:, sel2.float().argmax(1)] = True

    def seconda(m, z0, mem0):
        _, h2 = m(X2, p2[:, :-1], causal=True, z0=z0,
                  mem0=mem0)
        lo = testa(h2)
        lg = nn.functional.cross_entropy(
            lo[sel2], Y2[sel2]).item()
        lp = nn.functional.cross_entropy(
            lo[prima], Y2[prima]).item()
        return _ce(lo, Y2).item(), lg, lp

    def prova(nome, f=None, modo='porta'):
        m = copy.deepcopy(core).to(dev)
        if f:
            with torch.no_grad():
                f(m)
        m.eval()
        z1, h1 = m(X1, p1[:, :-1], causal=True)
        mem1 = m.annota(h1, sorpresa_simbolo(testa(h1), Y1),
                        p1[:, :-1])
        if modo == 'dimentica':
            z1, mem1 = None, None
        elif modo == 'mescola':
            z1 = z1.roll(1, 0)
            if mem1:
                mem1 = mem1[1:] + mem1[:1]
        elif modo == 'no_z':
            z1 = None
        elif modo == 'no_mem':
            mem1 = None
        l2, lg, lp = seconda(m, z1, mem1)
        print('%-42s %5.3f  %5.3f  %5.3f' % (nome, l2, lg, lp),
              flush=True)
        del m
        if dev == 'cuda':
            torch.cuda.empty_cache()

    _, h1 = core(X1, p1[:, :-1], causal=True)
    lo1 = testa(h1)
    print('finestra 1: loss %.3f  giunzioni %.3f (a caso %.3f)'
          % (_ce(lo1, Y1).item(),
             nn.functional.cross_entropy(
                 lo1[sel1], Y1[sel1]).item(),
             torch.log(torch.tensor(float(ngiun))).item()))
    print('giunzioni in finestra 2: %d a sequenza'
          % (sel2.sum(1).float().mean().item()))
    print('%-42s %5s  %9s  %s' % ('finestra 2', 'loss',
                                   'giunzioni', '1a giunz.'))
    prova('z portato dalla finestra 1')
    prova('tutto dimenticato', modo='dimentica')
    prova('solo z dimenticato (foglietto tenuto)',
          modo='no_z')
    prova('solo foglietto dimenticato (z tenuto)',
          modo='no_mem')
    prova('z+foglietto mescolati', modo='mescola')
    prova('senza comunicazione (lam=0)',
          lambda m: m.comm.lam.zero_())
    prova('senza foglietto (mem.lam=0)',
          lambda m: m.foglio.lam.zero_())
    prova('universi muti (zinj=0)',
          lambda m: m.zinj.weight.zero_())
    prova('lettura muta (zread=0)',
          lambda m: m.zread.out_proj.weight.zero_())
    d = core.zinj.out_features
    for k in range(core.K):
        prova('universo %d muto' % k,
              lambda m, k=k: m.zinj.weight[
                  :, k * d:(k + 1) * d].zero_())
    # chi ascolta chi sulle giunzioni di finestra 2
    z1, h1 = core(X1, p1[:, :-1], causal=True)
    mem1 = core.annota(h1, sorpresa_simbolo(testa(h1), Y1),
                       p1[:, :-1])
    z2, _ = core(X2, p2[:, :-1], causal=True, z0=z1,
                 mem0=mem1)
    _, w = core.comm(z2)
    print('chi ascolta chi in finestra 2:')
    for k in range(core.K):
        print('  universo %d ascolta' % k,
              [round(float(x), 3) for x in w.mean(0)[k]])


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    un_fatto = 'un' in sys.argv
    test = dati_dialogo(64, 2, un_fatto=un_fatto)
    train0 = dati_dialogo(8, 9, pos_modi=200,
                          un_fatto=un_fatto)
    vocab, n = vocab_da_sequenze(train0 + test)
    torch.manual_seed(0)
    core = MultiUniverse(n, K=K)
    # curriculum: i modi allontanano dal confine a fasi —
    # prima z deve conservarli per poco, poi sempre di piu'
    testa, hist = None, []
    for pos in (200, 150, 100, 50, 0):
        train = dati_dialogo(passi * 8, 1, pos_modi=pos,
                             un_fatto=un_fatto)
        h2, testa = allena(core, finestre(train, vocab, dev),
                           vocab, dev, passi=passi, batch=8,
                           testa=testa)
        hist += h2
        print('fase pos_modi=%d finita' % pos, flush=True)
    torch.save({'core': core.state_dict(),
                'testa': testa.state_dict(),
                'vocab': vocab},
               os.path.join(os.path.dirname(
                   os.path.abspath(__file__)),
                   'dati', 'universi_dialogo.pt'))
    print('train: %.3f -> %.3f' % (hist[0],
                                   sum(hist[-10:]) / 10))
    valuta(core, testa, finestre(test, vocab, dev), dev,
           vocab, un_fatto=un_fatto)
