"""La vigilanza esistente basta a rilevare la saturazione di d?

E2 (sonda_sviluppo) ha misurato un gap al plateau fra d=96 e
d=384: la mente piccola resta sopra. Domanda aperta: sotto quel
carico, l'organo di crescita che gia' esiste (vigilanza ->
cresci(), sull'asse K) si accorge del problema?

Tre misure su menti d=96 nate K=1, stesso compito saturante:

A) vigilanza derivata come in E1: mediana della seconda meta'
   del warmup. Se il plateau sta sotto la mediana di warmup —
   previsione: non scatta MAI. Misura il difetto strutturale
   della soglia assoluta: "sorpresa alta" non e' "imparo poco".

B) controllo: stessa mente senza vigilanza — la curva di
   riferimento.

C) segnale candidato post-hoc: pendenza della loss su finestra
   mobile. Non e' un meccanismo aggiunto: qui si misura solo se
   il segnale esiste ed e' separabile (pendenza in fase di
   apprendimento vs pendenza al plateau). Se i due regimi si
   separano, una vigilanza sullo stallo e' misurabile; se no,
   anche quella strada muore.

   python murlen/tests/sonda_vigilanza.py [cuda|cpu]"""
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import (vocab_da_sequenze,  # noqa: E402
                          impara_predizione)
from confronto_init import dati  # noqa: E402

PASSI, WARM, BLOCCO, FIN = 800, 40, 20, 40


def config(K, d=96):
    return dict(K=K, d=d, ctx=256, enc_layers=4, m=16,
                nhead=4, tempo=64)


def nascita(n, dev, **kw):
    torch.manual_seed(0)
    return MultiUniverse(n, **config(K=1, **kw))


def a_tratti(core, seqs, vocab, dev, passi, testa=None, **kw):
    """Allena a blocchi; ritorna (curva, hist_passo, testa)."""
    curva, hist = [], []
    fatti = 0
    while fatti < passi:
        h, testa = impara_predizione(
            core, seqs, vocab, passi=BLOCCO, device=dev,
            batch=1, testa=testa, **kw)
        hist += h
        fatti += BLOCCO
        curva.append((fatti, sum(h) / len(h), core.K))
    return curva, hist, testa


def stampa(etichetta, curva):
    print('%-14s %s' % (etichetta, '  '.join(
        '%.2f' % l for _, l, _ in curva)))
    print('%-14s K finale %d, crescite %d' % (
        '', curva[-1][2], curva[-1][2] - 1))


def pendenze(hist, fin):
    """Pendenza della loss su finestre mobili di `fin` passi."""
    return [hist[i] - hist[i - fin]
            for i in range(fin, len(hist))]


def main(dev):
    seqs = dati(600, 1)
    vocab, n = vocab_da_sequenze(seqs)

    # --- A) vigilanza assoluta derivata dal warmup ---
    print('=== A: vigilanza assoluta (soglia da warmup) ===')
    piccolo = nascita(n, dev)
    h0, testa = impara_predizione(
        piccolo, seqs, vocab, passi=WARM, device=dev, batch=1)
    vig = statistics.median(h0[-WARM // 2:])
    print('soglia: %.3f (mediana ultimi %d di warmup)' % (
        vig, WARM // 2))
    curva_a, _, testa = a_tratti(
        piccolo, seqs, vocab, dev, PASSI, testa=testa,
        vigilanza=vig, pazienza=20, universi_max=6)
    stampa('vigilante', curva_a)

    # --- B) controllo senza vigilanza ---
    print('=== B: controllo, nessuna vigilanza ===')
    ctrl = nascita(n, dev)
    curva_b, h, _ = a_tratti(ctrl, seqs, vocab, dev, PASSI)
    stampa('cieca', curva_b)

    # --- C) il segnale di stallo esiste? pendenza per regime ---
    print('=== C: segnale di stallo (pendenza su %d passi) ==='
          % FIN)
    p = pendenze(h, FIN)
    # regimi: prima meta' del training vs ultimo quarto
    meta, ultimo = p[:len(p) // 2], p[int(len(p) * .75):]
    qa = statistics.quantiles([abs(x) for x in meta], n=4)
    qb = statistics.quantiles([abs(x) for x in ultimo], n=4)
    q1a, q3a, q1b, q3b = qa[0], qa[2], qb[0], qb[2]
    print('|pendenza| prima meta\': Q1=%.4f Q3=%.4f  (n=%d)' % (
        q1a, q3a, len(meta)))
    print('|pendenza| ultimo quarto: Q1=%.4f Q3=%.4f  (n=%d)' % (
        q1b, q3b, len(ultimo)))
    print('loss ultimo quarto: mediana %.3f' %
          statistics.median(h[int(len(h) * .75):]))
    if q3b < q1a:
        print('-> regimi SEPARATI: pendenza al plateau sotto il '
              'Q1 della fase di apprendimento — un rilevatore '
              'di stallo e\' derivabile dalla distribuzione')
    else:
        print('-> regimi NON separati a questa finestra: la '
              'pendenza non distingue apprendimento da plateau')

    # --- verdetto ---
    print('=== verdetto ===')
    gap = curva_b[-1][1] - curva_a[-1][1]
    print('vigilanza assoluta: crescite=%d, differenza finale '
          'vs cieca=%.3f' % (curva_a[-1][2] - 1, gap))
    if curva_a[-1][2] == 1:
        print('MAI scattata: la soglia assoluta non distingue '
              '"imparo lento" da "sono saturo" — la loss fin '
              'sotto la mediana di warmup e ci resta')


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    main(dev)
