"""Replay nel sonno: il ciclo gateway+ipotalamo sul muro N=32.

sonda_scala_n32: a N=32 (~15 letture/fatto per episodio)
il lettore indirizzato resta piatto a 2000 passi su dati
freschi. Domanda: il ciclo veglia/sonno degli organi —
giorni vissuti in eval, notti con replay degli episodi
archiviati nel training — accumula esposizione per fatto
e accende il canale dove la singola passata fallisce?

A) controllo: 2000 passi su dati freschi (baseline piatto).
B) ciclo: 8 giorni da [veglia: 8 episodi vissuti in eval,
   mem0 portato, gateway.veglia sulle finestre] + [notte:
   dormi — episodi del giorno + replay dallo Store, passi
   divisi per numero di sequenze]. Stessi 2000 passi totali.

Se B << A sulle prime occorrenze -> l'esposizione cumulata
via replay basta; se entrambe piatte -> il vincolo e' la
densita' per episodio e il replay va ripensato.

python murlen/tests/sonda_replay.py [cuda|cpu]
"""
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import murlen.model as MM  # noqa: E402
from murlen.ponte import (vocab_da_sequenze,  # noqa: E402
                          tensorizza, TestaPredizione)
from murlen.store import Store  # noqa: E402
from murlen.gateway import Gateway  # noqa: E402
from universi_oltre_finestra import (  # noqa: E402
    K, allena, sorpresa_simbolo)
from sonda_pressione_indirizzo import (  # noqa: E402
    dati, lettore, GIUN, NVAL, WIN)

GIORNI, PASSI_GIORNO, EP_GIORNO = 8, 250, 8
PASSI_TOT = GIORNI * PASSI_GIORNO


def finestre(seqs, vocab, dev):
    """Come sonda_pressione_indirizzo.finestre ma sulla
    lunghezza propria delle sequenze (giorni 1024,
    replay ~256): niente finestre tutte padding."""
    L = max(len(s) for s in seqs)
    out = []
    for a in range(0, L, WIN):
        ids, pad = tensorizza(
            [s[a:a + WIN] for s in seqs], vocab, WIN)
        out.append((ids.to(dev), pad.to(dev)))
    return out


@torch.no_grad()
def veglia(core, gw, testa, seqs, vocab, dev, mem0=None):
    """Il giorno: finestre vissute in eval, il foglietto
    porta fra finestre (mem0), le finestre che scrivono
    tracce diventano episodi del giorno. -> mem finale."""
    mem = mem0
    for ids, pad in finestre(seqs, vocab, dev):
        X, Y = ids[:, :-1], ids[:, 1:]
        _, h = core(X, pad[:, :-1], causal=True,
                    mem0=mem)
        sorpresa = sorpresa_simbolo(testa(h), Y)
        mem = core.annota(h, sorpresa, pad[:, :-1])
        gw.veglia(X, sorpresa, pad[:, :-1],
                  core.foglio)
    return mem


def dormi(core, gw, testa, seqs, replay, vocab, dev,
          passi, batch=8):
    """La notte minima: prima gli episodi del giorno,
    poi il replay dallo Store; i passi si dividono
    per numero di sequenze — niente peso scelto."""
    n_g, n_r = len(seqs), len(replay)
    p_g = round(passi * n_g / (n_g + n_r)) if n_r \
        else passi
    hist = []
    if p_g:
        h, testa = allena(
            core, finestre(seqs, vocab, dev), vocab,
            dev, passi=p_g, batch=batch, testa=testa)
        hist += h
    if n_r and passi - p_g:
        h, testa = allena(
            core, finestre(replay, vocab, dev), vocab,
            dev, passi=passi - p_g, batch=batch,
            testa=testa)
        hist += h
    return hist, testa


def braccio(core_maker, vocab, dev):
    """A) controllo: una passata da 2000 passi freschi."""
    core = core_maker()
    test = dati(64, 2, 32)
    train = dati(PASSI_TOT * 8, 1, 32)
    hist, testa = allena(
        core, finestre(train, vocab, dev), vocab, dev,
        passi=PASSI_TOT, batch=8)
    print('A train: %.3f -> %.3f' % (
        hist[0], sum(hist[-10:]) / 10), flush=True)
    lettore(core, testa, test, vocab, dev)


def braccio_ciclo(core_maker, vocab, dev, tmp):
    """B) giorni di veglia + notti con replay."""
    core = core_maker().to(dev).eval()
    st = Store(str(tmp / 'episodi.jsonl'))
    gw = Gateway(st)
    test = dati(64, 2, 32)
    testa = TestaPredizione(
        max(vocab.values()) + 1,
        core.enc.tok.embedding_dim).to(dev)
    mem = None
    for g in range(GIORNI):
        giorno = dati(EP_GIORNO, 10 + g, 32)
        mem = veglia(core, gw, testa, giorno,
                     vocab, dev, mem0=mem)
        replay = gw.sonno()
        hist, testa = dormi(
            core, gw, testa, giorno, replay, vocab,
            dev, passi=PASSI_GIORNO)
        print('giorno %d: archivio=%d episodi, '
              'replay=%d, loss %.3f -> %.3f' % (
                  g, len(st.attive('episodio')),
                  len(replay), hist[0],
                  sum(hist[-10:]) / 10), flush=True)
    lettore(core, testa, test, vocab, dev)


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    tmp = tempfile.TemporaryDirectory()
    t0 = dati(8, 9, 32)
    tutti = [list(range(0, GIUN + NVAL + 2))]
    vocab, n = vocab_da_sequenze(
        t0 + dati(64, 2, 32) + tutti)

    def maker():
        torch.manual_seed(0)
        return MM.MultiUniverse(n, K=K)

    if len(sys.argv) < 3 or sys.argv[2] != 'b':
        print('=== A) controllo: 2000 passi freschi ===',
              flush=True)
        braccio(maker, vocab, dev)
    print('=== B) ciclo: %d giorni x %d passi ===' % (
        GIORNI, PASSI_GIORNO), flush=True)
    braccio_ciclo(maker, vocab, dev, Path(tmp.name))
