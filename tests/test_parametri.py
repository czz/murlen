"""Dimensioni del modello impostabili alla creazione."""
import pytest
import torch
import torch.nn as nn

from murlen.model import MultiUniverse, param_count
from murlen.ponte import vocab_da_sequenze, impara_predizione


def _teste(m):
    return {a.num_heads for a in m.modules()
            if isinstance(a, nn.MultiheadAttention)}


def test_valori_predefiniti_invariati():
    conti = [param_count(MultiUniverse(200, K=k, comm=c, init=i,
                                       tempo=t))
             for k, c, i, t in [(1, False, 'learned', 0),
                                (3, True, 'learned', 0),
                                (2, True, 'projected', 8)]]
    assert conti == [21481377, 33631588, 28875332]


def test_teste_impostabili():
    m = MultiUniverse(50, K=2, comm=True, init='projected',
                      enc_layers=1, d=32, m=4, nhead=4)
    assert _teste(m) == {4}


def test_contesto_impostabile():
    m = MultiUniverse(50, enc_layers=1, d=32, m=4, ctx=64)
    assert m.enc.pos.num_embeddings == 64


def test_d_non_divisibile_per_teste():
    with pytest.raises(ValueError):
        MultiUniverse(50, enc_layers=1, d=30, m=4, nhead=8)


def _due_passate(jitter):
    torch.manual_seed(0)
    m = MultiUniverse(50, K=2, comm=True, enc_layers=1, d=32, m=4,
                      jitter=jitter).train()
    ids = torch.randint(2, 50, (2, 10))
    pad = torch.zeros(2, 10, dtype=torch.bool)
    return m(ids, pad)[0], m(ids, pad)[0]


def test_jitter_impostabile():
    a, b = _due_passate(0.)
    assert torch.equal(a, b)
    a, b = _due_passate(0.1)
    assert not torch.equal(a, b)


def test_cresci_usa_i_parametri_del_modello():
    m = MultiUniverse(50, K=1, enc_layers=1, d=32, m=4, nhead=4,
                      jitter=0.)
    m.cresci()
    assert _teste(m) == {4}
    assert m.upd.blocks[-1].jitter == 0.


def test_impara_usa_il_contesto_del_modello():
    seqs = [list(range(2, 32))] * 3
    vocab, n = vocab_da_sequenze(seqs)
    core = MultiUniverse(n, enc_layers=1, d=32, m=4, ctx=16)
    hist, _ = impara_predizione(core, seqs, vocab, passi=2)
    assert len(hist) == 2


def test_contesto_piu_lungo_del_modello():
    seqs = [list(range(2, 32))] * 3
    vocab, n = vocab_da_sequenze(seqs)
    core = MultiUniverse(n, enc_layers=1, d=32, m=4, ctx=16)
    with pytest.raises(ValueError):
        impara_predizione(core, seqs, vocab, passi=1, ctx=64)


def test_piu_universi_si_parlano_e_imparano():
    m = MultiUniverse(50, K=3, enc_layers=1, d=32, m=4)
    assert m.comm_on and m.comm is not None
    assert m.tempo == 64


def test_un_universo_come_prima():
    m = MultiUniverse(50, K=1, enc_layers=1, d=32, m=4)
    assert not m.comm_on and m.comm is None
    assert m.tempo == 0


def test_valori_espliciti_rispettati():
    m = MultiUniverse(50, K=3, comm=False, tempo=0,
                      enc_layers=1, d=32, m=4)
    assert not m.comm_on and m.tempo == 0


def test_universi_allenati_di_default():
    torch.manual_seed(0)
    seqs = [[(i * 7 + j) % 20 + 2 for i in range(150)]
            for j in range(3)]
    vocab, n = vocab_da_sequenze(seqs)
    core = MultiUniverse(n, K=2, enc_layers=1, d=32, m=4)
    prima = [p.detach().clone() for p in core.upd.parameters()]
    impara_predizione(core, seqs, vocab, passi=2)
    assert any(not torch.equal(a, b) for a, b in
               zip(prima, core.upd.parameters()))


def test_foglietto_allenato_dentro_la_finestra():
    """Con tempo attivo il flusso consolida a ogni
    blocco: il foglietto impara anche dentro una
    finestra sola, senza bisogno di mem0."""
    torch.manual_seed(0)
    seqs = [[(i * 7 + j) % 20 + 2 for i in range(150)]
            for j in range(3)]
    vocab, n = vocab_da_sequenze(seqs)
    core = MultiUniverse(n, K=2, enc_layers=1, d=32, m=4)
    prima = [p.detach().clone() for p in
             core.foglio.parameters()]
    impara_predizione(core, seqs, vocab, passi=2)
    assert any(not torch.equal(a, b) for a, b in
               zip(prima, core.foglio.parameters()))


def _lunghe(n=100):
    ids = torch.randint(2, 50, (2, n))
    return ids, torch.zeros(2, n, dtype=torch.bool)


def test_cresci_da_uno_accende_comunicazione_e_tempo():
    torch.manual_seed(0)
    m = MultiUniverse(50, K=1, enc_layers=1, d=32, m=4).eval()
    ids, pad = _lunghe()
    _, h1 = m(ids, pad, causal=True)
    m.cresci()
    assert m.K == 2 and m.comm_on and m.tempo == 64
    assert float(m.comm.lam) == 0.
    z2, h2 = m.eval()(ids, pad, causal=True)
    assert z2.shape[1] == 2
    assert torch.allclose(h1, h2, atol=1e-5)


def test_cresci_rispetta_scelte_esplicite():
    m = MultiUniverse(50, K=1, comm=False, tempo=0,
                      enc_layers=1, d=32, m=4)
    m.cresci()
    assert m.K == 2 and not m.comm_on and m.tempo == 0


def test_cresci_da_uno_poi_impara():
    seqs = [[(i * 7 + j) % 20 + 2 for i in range(150)]
            for j in range(3)]
    vocab, n = vocab_da_sequenze(seqs)
    core = MultiUniverse(n, K=1, enc_layers=1, d=32, m=4)
    core.cresci()
    prima = float(core.comm.lam)
    impara_predizione(core, seqs, vocab, passi=3, lr=1e-2)
    assert float(core.comm.lam) != prima
