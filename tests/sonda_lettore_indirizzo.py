"""Il blocco e' l'indirizzamento o l'ambiguita'?

sonda_lettore_molti: giunzioni marca->risposta PIATTE a
ogni N (anche N=1) — mentre il simbolo-giunzione solo si
leggeva (densita). Due cause possibili, confondate:

1. INDIREZZAMENTO: la marca deve selezionare la traccia per
   contenuto — il canale non sa farlo
2. AMBIGUITA' DI RUOLO: MARK+q in testa e' seguita da ANS,
   in giunzione da GIUN — stesso simbolo, due risposte:
   il segnale si annulla da solo

Condizioni, N fatti in testa [MARK+i, ANS+v_i]:

- A qry: giunzioni [QRY+q, GIUN+v_q] — la query e' una
  famiglia DIVERSA dalla marca: indirizzata (serve trovare
  il fatto q per identita') ma senza ambiguita' di ruolo.
  Se nasce -> il blocco era l'ambiguita', non
  l'indirizzamento: recall associativo ESISTE
- B ciclo: giunzioni GIUN+v_q sole, ordine ciclico
  (k-esima -> fatto k mod N) — niente marca: indirizzo
  posizionale multi-fatto. Se nasce -> il canale legge
  contenuto indirizzato per posizione
- C come A, N=32: scala

python murlen/tests/sonda_lettore_indirizzo.py [cuda|cpu] [passi]
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
MARK, QRY, ANS, GIUN = 300, 340, 400, 450
CNTA, CNTB = 10, 20
L = 4 * WIN


def dati(n, seed, nfatti, usi, modo):
    """Testa di nfatti [MARK+i, ANS+v_i]; poi giunzioni
    uniformi sul resto della sequenza:
    modo 'qry'  -> [QRY+q, GIUN+v_q]
    modo 'ciclo'-> [GIUN+v_q] con q = indice ciclico."""
    r = random.Random(seed)
    out = []
    for _ in range(n):
        vals = [r.randrange(NVAL) for _ in range(nfatti)]
        s = []
        for i in range(nfatti):
            s += [MARK + i, ANS + vals[i]]
        query = [i for i in range(nfatti)
                 for _ in range(usi)]
        if modo == 'qry':
            r.shuffle(query)     # ordine libero: serve
                                 # l'identita' della query
        # 'ciclo': query resta ordinato — la posizione dice
        # quale fatto serve
        inizio = 2 * nfatti + 8
        spazi = (L - inizio) // max(1, len(query))
        ca, cb = r.randrange(8), r.randrange(8)
        qi = 0
        while len(s) < L:
            if qi < len(query) and \
                    len(s) >= inizio + spazi * qi:
                q = query[qi]
                s += ([QRY + q] if modo == 'qry' else []) \
                    + [GIUN + vals[q]]
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
    """CE sulle risposte GIUN+.. delle finestre >=1 per
    mem0 intatto / svuotato / episodio-1."""
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
    ('A qry   N=8', 8, 8, 'qry'),
    ('B ciclo N=8', 8, 8, 'ciclo'),
    ('C qry   N=32', 32, 8, 'qry'),
]

if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    for nome, nf, usi, modo in COND:
        print('=== %s ===' % nome, flush=True)
        test = dati(64, 2, nf, usi, modo)
        t0 = dati(8, 9, nf, usi, modo)
        tutti = [list(range(0, GIUN + NVAL + 2))]
        vocab, n = vocab_da_sequenze(t0 + test + tutti)
        torch.manual_seed(0)
        core = MM.MultiUniverse(n, K=K)
        train = dati(passi * 8, 1, nf, usi, modo)
        hist, testa = allena(
            core, finestre(train, vocab, dev), vocab, dev,
            passi=passi, batch=8)
        print('train: %.3f -> %.3f' % (
            hist[0], sum(hist[-10:]) / 10), flush=True)
        lettore(core, testa, test, vocab, dev)
        del core, testa
        if dev == 'cuda':
            torch.cuda.empty_cache()
