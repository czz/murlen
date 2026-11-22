import pytest
import json
import math
import os

import torch

from murlen.ponte import (vocab_da_sequenze, tensorizza,
                          impara_predizione, salva_sicuro,
                          carica_sicuro, PAD_ID)
from aiuti import modello, sequenze_ripetitive, sequenze_variabili

DATI = os.path.join(os.path.dirname(__file__), 'dati')


def test_vocabolario():
    vocab, n = vocab_da_sequenze([[7, 100], [100, 3]])
    assert vocab == {3: 2, 7: 3, 100: 4}
    assert n == 5


def test_tensorizza():
    vocab, _ = vocab_da_sequenze([[7, 8, 9]])
    ids, pad = tensorizza([[7, 8], [7, 8, 9]], vocab, ctx=5)
    assert ids.tolist() == [[2, 3, 1, 1, 1], [2, 3, 4, 1, 1]]
    assert pad.tolist() == [[False, False, True, True, True],
                            [False, False, False, True, True]]
    assert (ids[pad] == PAD_ID).all()


def test_la_loss_scende():
    seqs = sequenze_ripetitive()
    vocab, n = vocab_da_sequenze(seqs)
    core = modello(n)
    hist, _ = impara_predizione(core, seqs, vocab, passi=40,
                                ctx=64, lr=3e-3)
    assert all(math.isfinite(x) for x in hist)
    assert hist[-1] < hist[0] * 0.5


def test_tempo_con_padding_impara():
    seqs = sequenze_variabili()
    vocab, n = vocab_da_sequenze(seqs)
    core = modello(n, tempo=8)
    hist, _ = impara_predizione(core, seqs, vocab, passi=20,
                                ctx=64, lr=3e-3)
    assert all(math.isfinite(x) for x in hist)
    assert hist[-1] < hist[0]


def _storico_riferimento(tempo):
    seqs = sequenze_variabili()
    vocab, n = vocab_da_sequenze(seqs)
    core = modello(n, tempo=tempo)
    hist, _ = impara_predizione(core, seqs, vocab, passi=8,
                                ctx=64, lr=1e-3)
    return hist


def test_stesso_risultato_di_prima():
    """Con tempo=0 le ottimizzazioni non devono cambiare
    l'apprendimento: stesso storico della loss del codice
    originale (tests/dati/riferimento_ponte.json)."""
    with open(os.path.join(DATI, 'riferimento_ponte.json')) as f:
        rif = json.load(f)['tempo0']
    hist = _storico_riferimento(0)
    assert len(hist) == len(rif)
    for a, b in zip(hist, rif):
        assert abs(a - b) < 1e-4, (hist, rif)


def test_universi_saltati_quando_inutili():
    seqs = sequenze_variabili()
    vocab, n = vocab_da_sequenze(seqs)
    core = modello(n)
    chiamate = []
    core.upd.register_forward_hook(
        lambda *a: chiamate.append(1))
    impara_predizione(core, seqs, vocab, passi=2, ctx=64)
    assert chiamate == []
    core_t = modello(n, tempo=8)
    core_t.upd.register_forward_hook(
        lambda *a: chiamate.append(1))
    impara_predizione(core_t, seqs, vocab, passi=1, ctx=64)
    assert chiamate  # con tempo gli universi servono


def test_sequenze_troppo_corte_ignorate():
    """Sequenze di 0-1 simboli: niente da predire
    (X vuota), non entrano nel corpus — anche una
    riga tutta padding in mezzo ad altre non rompe."""
    seqs = sequenze_ripetitive() + [[], [7]]
    vocab, n = vocab_da_sequenze(seqs)
    hist, _ = impara_predizione(modello(n), seqs, vocab,
                                passi=3, ctx=64)
    assert len(hist) == 3


def test_solo_sequenze_corte_errore():
    vocab, _ = vocab_da_sequenze([[5]])
    with pytest.raises(ValueError):
        impara_predizione(modello(5), [[], [5]], vocab,
                          passi=1)


def test_mini_batch():
    seqs = sequenze_ripetitive() + sequenze_variabili()
    vocab, n = vocab_da_sequenze(seqs)
    core = modello(n)
    hist, _ = impara_predizione(core, seqs, vocab, passi=40,
                                ctx=64, lr=3e-3, batch=2)
    assert len(hist) == 40
    assert all(math.isfinite(x) for x in hist)
    assert sum(hist[-5:]) / 5 < sum(hist[:5]) / 5



def test_batch_pari_a_tutte_uguale_a_none():
    seqs = sequenze_variabili()
    vocab, n = vocab_da_sequenze(seqs)
    h1, _ = impara_predizione(modello(n), seqs, vocab,
                              passi=3, ctx=64)
    h2, _ = impara_predizione(modello(n), seqs, vocab,
                              passi=3, ctx=64, batch=len(seqs))
    assert h1 == h2


def _lr_per_passo(monkeypatch, **kw):
    import murlen.ponte as ponte
    lrs, orig = [], ponte._passo

    def spia(core, testa, opt, *a):
        lrs.append(opt.param_groups[0]['lr'])
        return orig(core, testa, opt, *a)
    monkeypatch.setattr(ponte, '_passo', spia)
    seqs = sequenze_ripetitive()
    vocab, n = vocab_da_sequenze(seqs)
    impara_predizione(modello(n), seqs, vocab, ctx=64, lr=1e-3, **kw)
    return lrs


def test_riscaldamento_lr_sale(monkeypatch):
    lrs = _lr_per_passo(monkeypatch, passi=8, riscaldamento=4)
    attesi = [2.5e-4, 5e-4, 7.5e-4, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
    assert all(abs(a - b) < 1e-9 for a, b in zip(lrs, attesi)), lrs


def test_senza_riscaldamento(monkeypatch):
    lrs = _lr_per_passo(monkeypatch, passi=5, riscaldamento=0)
    assert lrs == [1e-3] * 5


def test_riscaldamento_automatico(monkeypatch):
    lrs = _lr_per_passo(monkeypatch, passi=20)
    assert abs(lrs[0] - 2.5e-4) < 1e-9 and lrs[-1] == 1e-3


def test_testa_continua_fra_chiamate():
    seqs = sequenze_ripetitive()
    vocab, n = vocab_da_sequenze(seqs)
    core = modello(n)
    h1, testa = impara_predizione(core, seqs, vocab, passi=30,
                                  ctx=64, lr=3e-3)
    peso = testa.lin.weight.detach().clone()
    h2, testa2 = impara_predizione(core, seqs, vocab, passi=3,
                                   ctx=64, lr=3e-3, testa=testa)
    assert testa2 is testa
    assert not torch.equal(peso, testa.lin.weight)
    # la testa non riparte da zero: niente risalita della loss
    assert h2[0] < h1[0] / 2


def test_testa_troppo_piccola():
    seqs = sequenze_ripetitive()
    vocab, n = vocab_da_sequenze(seqs)
    from murlen.ponte import TestaPredizione
    with pytest.raises(ValueError):
        impara_predizione(modello(n), seqs, vocab, passi=1, ctx=64,
                          testa=TestaPredizione(3, 32))


def test_checkpoint_roundtrip(tmp_path):
    core = modello(10, seed=1)
    p = str(tmp_path / 'm.pt')
    assert salva_sicuro(core, p)
    altro = modello(10, seed=2)
    assert carica_sicuro(altro, p) is not None
    for a, b in zip(core.parameters(),
                    altro.parameters()):
        assert torch.equal(a, b)


def test_checkpoint_rifiuta_nan(tmp_path):
    core = modello(10)
    p = str(tmp_path / 'm.pt')
    assert salva_sicuro(core, p)
    with torch.no_grad():
        next(core.parameters()).fill_(float('nan'))
    assert not salva_sicuro(core, p)
    # il sano sul disco non e' stato toccato
    altro = modello(10, seed=2)
    assert carica_sicuro(altro, p) is not None
    assert not os.path.exists(p + '.prev')


def test_checkpoint_prev_ripiega(tmp_path):
    core = modello(10, seed=1)
    p = str(tmp_path / 'm.pt')
    assert salva_sicuro(core, p)
    primo = {k: v.clone() for k, v in
             core.state_dict().items()}
    with torch.no_grad():
        for q in core.parameters():
            q.add_(1.)
    assert salva_sicuro(core, p)
    assert os.path.exists(p + '.prev')
    # il principale si corrompe: il .prev salva
    with open(p, 'wb') as f:
        f.write(b'rotto')
    altro = modello(10, seed=2)
    assert carica_sicuro(altro, p) is not None
    sd = altro.state_dict()
    for k, v in primo.items():
        assert torch.equal(sd[k], v)


def test_checkpoint_con_z_e_mem(tmp_path):
    core = modello(10)
    p = str(tmp_path / 'm.pt')
    z = torch.randn(1, 2, 4, 32)
    mem = [torch.randn(3, 32)]
    assert salva_sicuro(core, p, z=z, mem=mem)
    sd = carica_sicuro(modello(10), p)
    assert torch.equal(sd['z'], z)
    assert torch.equal(sd['mem'][0], mem[0])
