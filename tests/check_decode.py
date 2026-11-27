"""L'ultimo anello: il valore recuperato e' decodificabile?

Catena coppia (sonda_coppia.txt): copertura ok, chiavi ok,
gate aperto (lam~1), il lettore legge (prima 1.95-2.85 vs
caso 2.996) ma non in profondita'. Resta da capire dove
finisce il margine:

- se l'attenzione recupera la traccia giusta ma la testa
  non estrae v -> collo in DECODIFICA (valore h troppo
  povero/contestuale);
- se nemmeno il valore perfetto basta -> collo nella
  STRUTTURA della traccia (h[ANS] non porta v usabile).

ORACOLO: replay delle finestre con _leggi_foglietto
patchato: alle posizioni X[i]=QRY+q l'attenzione e'
one-hot sulla riga la cui chiave ha meta' sinistra
emb(MARK+q) — indirizzamento perfetto per costruzione.
'prima' qui e' il soffitto della decodifica.

python murlen/tests/check_decode.py [cuda|cpu] [passi]
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import murlen.model as MM  # noqa: E402
from murlen.ponte import vocab_da_sequenze  # noqa: E402
from universi_oltre_finestra import K, allena  # noqa: E402
from sonda_pressione_indirizzo import (  # noqa: E402
    dati, finestre, lettore, MARK, QRY, GIUN, NVAL,
    NFATTI_MAX)

NFATTI = 32


@torch.no_grad()
def oracolo(core, testa, test, vocab, dev, bersaglio,
            nome):
    """prima con lettura perfetta: per ogni QRY+q si
    inietta il valore della riga del buffer la cui
    meta' sinistra di chiave e' emb(bersaglio+q) —
    'mark' = traccia del binding originale,
    'query' = traccia di un'occorrenza precedente
    della stessa coppia query,
    'emb' = emb(GIUN+v) del fatto vero: il valore
    pulito, senza passare dal buffer."""
    core.eval()
    W = finestre(test, vocab, dev)
    giun = torch.tensor([vocab[GIUN + i]
                         for i in range(NVAL)], device=dev)
    qry = torch.tensor([vocab[QRY + i]
                        for i in range(NFATTI)],
                       device=dev)
    emb = core.enc.tok.weight
    z, mem, stati = None, None, []
    for ids, pad in W:
        stati.append((z, mem))
        z, h = core(ids[:, :-1].to(dev),
                    pad[:, :-1].to(dev), causal=True,
                    z0=z, mem0=mem)
        mem = core.annota(h, None, pad[:, :-1].to(dev))
    d = h.size(2)

    if nome == 'emb':
        # il valore vero di ogni fatto: MARK+q e'
        # seguito da GIUN+v nella sequenza — lo
        # leggo direttamente, niente buffer
        v_of = [dict() for _ in range(W[0][0].size(0))]
        for ids, pad in W:
            for b in range(ids.size(0)):
                for q in range(NFATTI):
                    w = (ids[b] == vocab[MARK + q]
                         ).nonzero().flatten()
                    if len(w) and w[0] + 1 < ids.size(1):
                        v_of[b].setdefault(
                            q, ids[b, w[0] + 1].item())
        righe_bers = [None] * len(stati)
        media = [sum(len(v) for v in v_of)
                 / len(v_of)]
    else:
        # riga del bersaglio per fatto nel buffer a ogni
        # finestra: meta' sinistra della chiave decodata
        # al simbolo piu' vicino
        righe_bers = []
        for z_in, mem_in in stati:
            righe = []
            if mem_in is None:
                righe_bers.append(None)
                continue
            for b in range(len(mem_in)):
                m = {}
                if mem_in[b].size(0):
                    sim = (mem_in[b][:, :d].unsqueeze(1) -
                           emb.unsqueeze(0)
                           ).norm(dim=2).argmin(1)
                    for q in range(NFATTI):
                        w = (sim == bersaglio[q]
                             ).nonzero().flatten()
                        if len(w):
                            m[q] = w[0].item()
                righe.append(m)
            righe_bers.append(righe)
    # auto-verifica: quante righe bersaglio per episodio
    # e quante iniezioni — se l'oracolo non inietta, il
    # risultato e' falso, non "negativo"
    media = [sum(len(r) for r in rr) / max(len(rr), 1)
             for rr in righe_bers if rr] \
        if nome != 'emb' else media
    iniez = [0]

    Xb = [None]
    blk = [0]

    def leggi_oracolo(h_j, coppie=None):
        B, T, _ = h_j.shape
        j = blk[0]
        blk[0] += 1
        mem_loc = core._mem
        n = [m.size(0) for m in mem_loc]
        S = max(n)
        if not S:
            return h_j
        vv = h_j.new_zeros(B, S, d)
        for b in range(B):
            if n[b]:
                vv[b, :n[b]] = mem_loc[b][:, 2 * d:]
        att = h_j.new_zeros(B, T, S)
        rm = righe_bers[wi_or[0]]
        if nome == 'emb':
            for b in range(B):
                for i in range(T):
                    hit = (Xb[0][b, j * core.tempo + i]
                           == qry).nonzero()
                    if len(hit):
                        q = hit[0].item()
                        if q in v_of[b]:
                            h_j[b, i] = h_j[b, i] + \
                                core.foglio.lam * \
                                emb[v_of[b][q]]
                            iniez[0] += 1
        elif rm is not None:
            for b in range(B):
                for i in range(T):
                    hit = (Xb[0][b, j * core.tempo + i]
                           == qry).nonzero()
                    if len(hit):
                        q = hit[0].item()
                        if q in rm[b]:
                            att[b, i, rm[b][q]] = 1.
                            iniez[0] += 1
        return h_j + core.foglio.lam * (att @ vv)

    wi_or = [0]
    orig = core._leggi_foglietto
    core._leggi_foglietto = leggi_oracolo
    try:
        tot, cnt = 0., 0
        for wi, (ids, pad) in enumerate(W):
            if wi == 0:
                continue
            wi_or[0] = wi
            X = ids[:, :-1].to(dev)
            Y = ids[:, 1:].to(dev)
            z_in, mem_in = stati[wi]
            Xb[0] = X
            blk[0] = 0
            _, h = core(X, pad[:, :-1].to(dev), causal=True,
                        z0=z_in, mem0=mem_in)
            sel = torch.isin(Y, giun)
            prima = torch.zeros_like(sel)
            B, T = X.shape
            for b in range(B):
                visti = set()
                for i in range(T):
                    hit = (X[b, i] == qry).nonzero()
                    if len(hit):
                        q = hit[0].item()
                        if sel[b, i] and q not in visti:
                            prima[b, i] = True
                        visti.add(q)
            lo = testa(h)
            if prima.any():
                tot += nn.functional.cross_entropy(
                    lo[prima], Y[prima]).item() \
                    * prima.sum().item()
                cnt += prima.sum().item()
        print('oracolo-%-5s prima: %.3f  (n=%d, a caso'
              ' %.3f)  [righe/episodio %s, iniezioni %d]'
              % (nome, tot / max(cnt, 1), cnt,
                 torch.log(torch.tensor(float(NVAL))).item(),
                 ['%.0f' % m for m in media], iniez[0]),
              flush=True)
    finally:
        core._leggi_foglietto = orig


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    seed = next((int(a[4:]) for a in sys.argv
                 if a.startswith('seed')), 0)
    NFATTI = next((int(a[1:]) for a in sys.argv
                   if a[:1] == 'n' and a[1:].isdigit()),
                  NFATTI)
    cap = next((int(a[3:]) for a in sys.argv
                if a.startswith('cap')), None)
    aperto = next((int(a[6:]) for a in sys.argv
                   if a.startswith('aperto')), 0)
    blocchi = None
    if NFATTI > NFATTI_MAX:
        # i range di default coprono 32 fatti: oltre
        # le query colliderebbero con ANS e GIUN —
        # si allocano blocchi disgiunti
        MARK = 1000
        QRY = MARK + NFATTI + 10
        ANS = QRY + NFATTI + 10
        GIUN = ANS + NVAL + 10
        blocchi = (MARK, QRY, ANS, GIUN)
    test = dati(64, 2 + seed * 1000, NFATTI, blocchi)
    t0 = dati(8, 9 + seed * 1000, NFATTI, blocchi)
    tutti = [list(range(0, GIUN + NVAL + 2))]
    vocab, n = vocab_da_sequenze(t0 + test + tutti)
    torch.manual_seed(seed)
    core = MM.MultiUniverse(
        n, K=K, coppia=True, simbolo='sim' in sys.argv,
        mem_cap=cap)
    if 'curr' in sys.argv:
        # curriculum su prefisso: la fase con n_ep=k e'
        # identica al compito a N=k (marche ordinate) —
        # il canale si forma dove il task e' noto-formabile
        # poi si estende a marche nuove con gradiente continuo
        ramp = [NFATTI // 8, NFATTI // 4, NFATTI // 2,
                NFATTI]
        pf = passi // len(ramp)
        hist, testa = [], None
        for fi, n_ep in enumerate(ramp):
            print('=== fase %d/%d: n_ep=%d, %d passi ==='
                  % (fi + 1, len(ramp), n_ep, pf),
                  flush=True)
            tr = dati(pf * 8, 1 + seed * 1000 + fi,
                      NFATTI, blocchi, n_ep=n_ep,
                      pref=True)
            h_f, testa = allena(
                core, finestre(tr, vocab, dev), vocab,
                dev, passi=pf, batch=8, seed=seed + fi,
                testa=testa, lam_aperto=aperto)
            hist += h_f
            del tr
            if dev == 'cuda':
                torch.cuda.empty_cache()
    else:
        train = dati(passi * 8, 1 + seed * 1000, NFATTI,
                     blocchi)
        hist, testa = allena(
            core, finestre(train, vocab, dev), vocab, dev,
            passi=passi, batch=8, seed=seed,
            lam_aperto=aperto)
    print('train: %.3f -> %.3f   lam=%.3f   cap=%d'
          '   aperto=%d' % (
              hist[0], sum(hist[-10:]) / 10,
              core.foglio.lam.item(), core._mem_cap,
              aperto), flush=True)
    lettore(core, testa, test, vocab, dev, blocchi,
            nq=NFATTI)
    for nome, b_ in (('mark', MARK), ('query', QRY),
                     ('emb', None)):
        bersaglio = None if b_ is None else \
            torch.tensor(
                [vocab[b_ + i] for i in range(NFATTI)],
                device=dev)
        oracolo(core, testa, test, vocab, dev,
                bersaglio, nome)
