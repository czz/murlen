"""Perche' il lettore nasce nel dialogo e non nell'esame?

Fatti misurati: su un_fatto (1 fatto in testa, giunzioni che
lo riusano nel flusso) il foglietto porta il fatto (1.27 <
caso 1.39, foglietto solo); sui compiti scrivi-poi-esame e
sul libro non si forma mai (ablazioni piatte).

Qui si parte dallo scheletro positivo e si toglie UN fattore
alla volta — mente fresca per condizione, stessa ricetta di
training (5 fasi pos_modi, 100 passi/fase), poi l'ablazione
intatto / svuotato / tracce di altro episodio sulla prima
giunzione di finestra 2:

- F0 copia/uso-fitto/curriculum: il positivo noto (controllo)
- F1 tabella arbitraria: la testa dichiara [nome, valore] e
  la giunzione emette il valore — casuale a episodio, non
  imparabile nei pesi: serve per forza la traccia
- F2 uso singolo: una sola giunzione in finestra 2 — il
  lookup e' allenato una volta per episodio, come l'esame
- F3 senza curriculum: pos_modi=0 subito — i fatti stanno
  sempre lontano dal confine
- F4 tabella + uso singolo: la struttura-esame dentro il
  flusso — atteso negativo

Lettura: la condizione che uccide il positivo dice QUALE
ingrediente e' necessario alla nascita del lettore.

python murlen/tests/sonda_lettore_fattori.py [cuda|cpu] [passi]
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
NMOD, NVAL = 4, 16
MA, ANS, GIUN = 300, 310, 400
CNTA, CNTB = 10, 20
JUN_OGNI, JUN_FASE = 8, 5
JUN_POS_UNA = 400     # F2/F4: la giunzione singola sta in
                      # finestra 2 (posizione assoluta)


def dati(n, seed, pos_modi, tabella=False, usa_una=False,
         L=2 * WIN):
    """un_fatto esteso: testa [30]*pos_modi + fatto, poi coppie
    alternate di contatori; ogni JUN_OGNI coppie il flusso A
    emette la giunzione.
    tabella=False: fatto [MA+a], giunzione GIUN+a (copia,
      binding fisso a ogni episodio).
    tabella=True: fatto [MA+a, ANS+v] con v casuale,
      giunzione GIUN+v (binding arbitrario: la risposta
      sta solo nella traccia del fatto)."""
    r = random.Random(seed)
    out = []
    for _ in range(n):
        a = r.randrange(NMOD)
        v = r.randrange(NVAL)
        fatto = [MA + a, ANS + v] if tabella else [MA + a]
        s = [30] * pos_modi + fatto
        ca, cb = r.randrange(8), r.randrange(8)
        p, fatta = 0, False
        while len(s) < L:
            if (usa_una and not fatta and
                    len(s) >= JUN_POS_UNA) or (
                    not usa_una and p % JUN_OGNI == JUN_FASE):
                fatta = True
                s.append(GIUN + (v if tabella else a))
            else:
                s.append(CNTA + ca % 8)
            ca += 1
            s.append(CNTB + cb % 8)
            cb += 1
            p += 1
        out.append(s[:L])
    return out


def finestre(seqs, vocab, dev):
    out = []
    for a in range(0, len(seqs[0]), WIN):
        ids, pad = tensorizza(
            [s[a:a + WIN] for s in seqs], vocab, WIN)
        out.append((ids.to(dev), pad.to(dev)))
    return out


@torch.no_grad()
def lettore(core, testa, test, vocab, dev, tabella):
    """Ablazione sulle giunzioni di finestra 2: il foglietto
    intatto aiuta? e conta il contenuto (tracce di un altro
    episodio)?"""
    core.eval()
    ids1, p1 = finestre(test, vocab, dev)[0]
    ids2, p2 = finestre(test, vocab, dev)[1]
    X1, Y1 = ids1[:, :-1], ids1[:, 1:]
    X2, Y2 = ids2[:, :-1], ids2[:, 1:]
    ng = NVAL if tabella else NMOD
    giun = torch.tensor([vocab[GIUN + i] for i in range(ng)],
                        device=dev)
    sel2 = torch.isin(Y2, giun)
    prima = torch.zeros_like(sel2)
    prima[:, sel2.float().argmax(1)] = True
    caso = torch.log(torch.tensor(float(ng))).item()

    z1, h1 = core(X1, p1[:, :-1], causal=True)
    mem = core.annota(h1, sorpresa_simbolo(testa(h1), Y1),
                      p1[:, :-1])
    B = len(mem)
    cond = {
        'intatto': mem,
        'svuotato': None,
        'episodio-1': [mem[(b + 1) % B] for b in range(B)],
    }
    for nome, m0 in cond.items():
        _, h2 = core(X2, p2[:, :-1], causal=True, z0=z1,
                     mem0=m0)
        lo = testa(h2)
        lg = nn.functional.cross_entropy(
            lo[sel2], Y2[sel2]).item()
        lp = nn.functional.cross_entropy(
            lo[prima], Y2[prima]).item()
        print('lettore %-10s giunzioni %.3f  1a %.3f'
              '  (a caso %.3f)' % (nome, lg, lp, caso),
              flush=True)


COND = [
    # nome, tabella, usa_una, curriculum
    ('F0 copia/uso-fitto/curr', False, False, True),
    ('F1 tabella/uso-fitto/curr', True, False, True),
    ('F2 copia/uso-singolo/curr', False, True, True),
    ('F3 copia/uso-fitto/no-curr', False, False, False),
    ('F4 tabella/uso-singolo/curr', True, True, True),
]

if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    fasi = (200, 150, 100, 50, 0)
    for nome, tabella, usa_una, curr in COND:
        print('=== %s ===' % nome, flush=True)
        test = dati(64, 2, 0, tabella, usa_una)
        t0 = dati(8, 9, fasi[0], tabella, usa_una)
        vocab, n = vocab_da_sequenze(t0 + test)
        torch.manual_seed(0)
        core = MM.MultiUniverse(n, K=K)
        testa, hist = None, []
        # stessi passi totali in ogni condizione: senza
        # curriculum le fasi restano tutte a pos_modi=0
        for pos in (fasi if curr else (0,) * len(fasi)):
            train = dati(passi * 8, 1, pos, tabella, usa_una)
            h2, testa = allena(
                core, finestre(train, vocab, dev), vocab,
                dev, passi=passi, batch=8, testa=testa)
            hist += h2
        print('train: %.3f -> %.3f' % (
            hist[0], sum(hist[-10:]) / 10), flush=True)
        lettore(core, testa, test, vocab, dev, tabella)
        del core, testa
        if dev == 'cuda':
            torch.cuda.empty_cache()
