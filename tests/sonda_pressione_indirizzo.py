"""Pressione alta sull'indirizzamento.

sonda_lettore_indirizzo: giunzioni QRY->GIUN erano il 6%
del flusso (contatori dominavano la loss) — il gradiente
sul canale di ricerca era debole. Qui il flusso dopo la
testa e' SOLO coppie query->risposta: ~50% dei simboli
sono risposte da recuperare per identita'. Se il canale
non nasce nemmeno sotto questa pressione, il limite e'
dell'organo, non del segnale.

Ablazioni sulle giunzioni delle finestre >=1:
intatto / svuotato / episodio-1 / senza-z — il quarto
dice se era z (memoria di lavoro) a portare la tabella.

python murlen/tests/sonda_pressione_indirizzo.py [cuda|cpu] [passi]
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
L = 4 * WIN


def dati(n, seed, nfatti, simboli=None, n_ep=None,
         pref=False):
    """Testa [MARK+i, ANS+v_i] poi il resto della
    sequenza e' solo coppie [QRY+q, GIUN+v_q]
    mischiate — ogni fatto interrogato ~(L-2N)/(2N)
    volte. simboli = (mark, qry, ans, giun) per
    nfatti oltre i range di default. n_ep<nfatti:
    ogni episodio usa solo n_ep fatti — scelti a
    caso fra gli nfatti, oppure il prefisso 0..n_ep-1
    (pref=True: il task della fase e' identico al
    compito a N=n_ep, marche in ordine)."""
    mk, qy, an, gj = simboli or (MARK, QRY, ANS, GIUN)
    r = random.Random(seed)
    out = []
    for _ in range(n):
        if n_ep is None or n_ep >= nfatti:
            ids_f = list(range(nfatti))
        elif pref:
            ids_f = list(range(n_ep))
        else:
            ids_f = r.sample(range(nfatti), n_ep)
        vals = {i: r.randrange(NVAL) for i in ids_f}
        s = []
        for i in ids_f:
            s += [mk + i, an + vals[i]]
        npair = (L - len(s)) // 2
        for i in range(npair):
            q = ids_f[i % len(ids_f)]  # ordine ciclico:
                                       # copertura uniforme
            s += [qy + q, gj + vals[q]]
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
def lettore(core, testa, test, vocab, dev,
            simboli=None, nq=NFATTI_MAX):
    """CE sulle risposte GIUN+.. per ablazioni:
    intatto / svuotato / episodio-1 / senza-z.
    La maschera 'prima' tiene solo la prima occorrenza
    di ogni fatto nella finestra: l'unica posizione
    dove la coppia non e' ancora apparsa in-window e
    la risposta puo' venire solo da z o foglietto —
    il resto lo risolve l'induzione sul contesto."""
    core.eval()
    gj = simboli[3] if simboli else GIUN
    qy = simboli[1] if simboli else QRY
    giun = torch.tensor([vocab[gj + i]
                         for i in range(NVAL)], device=dev)
    qry = torch.tensor([vocab[qy + i]
                        for i in range(nq)],
                       device=dev)
    W = finestre(test, vocab, dev)
    z, mem, stati = None, None, []
    for ids, pad in W:
        X, Y = ids[:, :-1].to(dev), ids[:, 1:].to(dev)
        stati.append((z, mem))
        z, h = core(X, pad[:, :-1].to(dev), causal=True,
                    z0=z, mem0=mem)
        mem = core.annota(h, sorpresa_simbolo(testa(h), Y),
                          pad[:, :-1].to(dev))

    def maschere(X, Y):
        """Risposta GIUN e sottoinsieme 'prima'
        occorrenza del fatto nella finestra."""
        sel = torch.isin(Y, giun)
        prima = torch.zeros_like(sel)
        B, T = X.shape
        for b in range(B):
            visti = torch.zeros(len(qry), dtype=torch.bool,
                              device=dev)
            xb = X[b]
            for i in range(T):
                hit = (xb[i] == qry).nonzero()
                if len(hit):
                    q = hit[0].item()
                    if sel[b, i] and not visti[q]:
                        prima[b, i] = True
                    visti[q] = True
        return sel, prima

    for nome in ('intatto', 'svuotato', 'episodio-1',
                 'senza-z'):
        tot = {'giunzioni': 0., 'prima': 0.}
        cnt = {'giunzioni': 0, 'prima': 0}
        for wi, (ids, pad) in enumerate(W):
            if wi == 0:
                continue
            X, Y = ids[:, :-1].to(dev), ids[:, 1:].to(dev)
            z_in, mem_in = stati[wi]
            if nome == 'senza-z':
                z_in = None
            elif nome == 'svuotato':
                mem_in = None
            elif nome == 'episodio-1':
                mem_in = [mem_in[(b + 1) % len(mem_in)]
                          for b in range(len(mem_in))]
            _, h = core(X, pad[:, :-1].to(dev), causal=True,
                        z0=z_in, mem0=mem_in)
            lo = testa(h)
            sel_all, sel_prima = maschere(X, Y)
            for et, sel in (('giunzioni', sel_all),
                            ('prima', sel_prima)):
                if sel.any():
                    tot[et] += nn.functional.cross_entropy(
                        lo[sel], Y[sel]).item() \
                        * sel.sum().item()
                    cnt[et] += sel.sum().item()
        caso = torch.log(torch.tensor(float(NVAL))).item()
        print('lettore %-10s giunzioni %.3f  prima %.3f'
              '  (n=%d/%d, a caso %.3f)' % (
                  nome,
                  tot['giunzioni'] / max(cnt['giunzioni'], 1),
                  tot['prima'] / max(cnt['prima'], 1),
                  cnt['giunzioni'], cnt['prima'], caso),
              flush=True)


COND = [('N=8', 8), ('N=32', 32)]

if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    for nome, nf in COND:
        print('=== %s ===' % nome, flush=True)
        test = dati(64, 2, nf)
        t0 = dati(8, 9, nf)
        tutti = [list(range(0, GIUN + NVAL + 2))]
        vocab, n = vocab_da_sequenze(t0 + test + tutti)
        torch.manual_seed(0)
        core = MM.MultiUniverse(n, K=K)
        train = dati(passi * 8, 1, nf)
        hist, testa = allena(
            core, finestre(train, vocab, dev), vocab, dev,
            passi=passi, batch=8)
        print('train: %.3f -> %.3f' % (
            hist[0], sum(hist[-10:]) / 10), flush=True)
        lettore(core, testa, test, vocab, dev)
        del core, testa
        if dev == 'cuda':
            torch.cuda.empty_cache()
