import torch

from murlen.model import MultiUniverse, param_count
from aiuti import modello, D_PICCOLO, M_PICCOLO


def _ids(B, N, vocab=20, pad_da=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, vocab, (B, N), generator=g)
    pad = torch.zeros(B, N, dtype=torch.bool)
    if pad_da is not None:
        for b, n in enumerate(pad_da):
            pad[b, n:] = True
            ids[b, n:] = 1
    return ids, pad


def test_forme_base():
    for kw in (dict(K=1, comm=False),
               dict(K=3, comm=True),
               dict(K=2, init='projected')):
        m = modello(20, **kw).eval()
        ids, pad = _ids(2, 12)
        z, h = m(ids, pad)
        assert z.shape == (2, m.K, M_PICCOLO, D_PICCOLO)
        assert h.shape == (2, 12, D_PICCOLO)


def test_forme_tempo():
    m = modello(20, tempo=4).eval()
    ids, pad = _ids(2, 12)
    z, h = m(ids, pad, causal=True)
    assert z.shape == (2, 2, M_PICCOLO, D_PICCOLO)
    assert h.shape == (2, 12, D_PICCOLO)


def test_causale_senza_tempo():
    m = modello(20).eval()
    ids, pad = _ids(1, 10)
    _, h1 = m(ids, pad, causal=True)
    ids2 = ids.clone()
    ids2[0, 6:] = 2  # cambia solo il futuro
    _, h2 = m(ids2, pad, causal=True)
    assert torch.allclose(h1[:, :6], h2[:, :6], atol=1e-5)


def test_causale_con_tempo():
    m = modello(20, tempo=4).eval()
    with torch.no_grad():
        m.zinj.weight.normal_(0, 0.1)  # universi "parlanti"
    ids, pad = _ids(1, 12)
    _, h1 = m(ids, pad, causal=True)
    ids2 = ids.clone()
    ids2[0, 8:] = 2  # cambia solo l'ultimo blocco
    _, h2 = m(ids2, pad, causal=True)
    assert torch.allclose(h1[:, :8], h2[:, :8], atol=1e-5)


def test_mem0_malformato_scartato():
    """mem0 con lunghezza o forma sbagliata viene
    scartato: il forward riparte da vuoto invece di
    andare in IndexError — stessa regola di z0."""
    m = modello(20, tempo=4).eval()
    ids, pad = _ids(2, 12)
    _, h1 = m(ids, pad, causal=True)
    for mem0 in ([torch.zeros(3, D_PICCOLO)],
                 [torch.zeros(0, D_PICCOLO)] * 99,
                 [torch.zeros(2, 2, D_PICCOLO)] * 2):
        _, h2 = m(ids, pad, causal=True, mem0=mem0)
        assert torch.allclose(h1, h2, atol=1e-5)


def test_core_mem_accessor():
    """core.mem e' l'accessor pubblico del
    foglietto: a finestra coincide col ritorno
    di annota; a blocchi (dove il flusso annota
    dentro forward) e' il solo modo di leggere
    lo stato — e il contenuto si ripassa come
    mem0."""
    m = modello(20, tempo=8).eval()
    # finestra (N <= tempo): annota la chiama
    # il chiamante, il ritorno e' lo stato
    ids, pad = _ids(2, 3)
    with torch.no_grad():
        _, h = m(ids, pad, causal=True)
        ret = m.annota(h, torch.rand(2, 3), pad)
    assert ret is m.mem
    # blocchi (N > tempo): il flusso annota a
    # ogni blocco, lo stato si legge da mem
    ids, pad = _ids(2, 24)
    n_scritte = [0]

    def scrivi(a, h_j, p_j):
        n_scritte[0] += 1
        # sorpresa con un picco netto: il fence
        # dei quartili lo seleziona di sicuro
        s = torch.zeros(h_j.size(0), h_j.size(1))
        s[:, -1] = 1e6
        return s
    with torch.no_grad():
        m(ids, pad, causal=True, scrivi=scrivi)
    assert n_scritte[0] == 3        # 12/4 blocchi
    assert all(t.dim() == 2 and
               t.size(1) == D_PICCOLO
               for t in m.mem)
    # posizione 0 sempre tracciata: ogni blocco
    # lascia almeno una traccia per elemento
    assert all(t.size(0) >= 3 for t in m.mem)
    # e il foglietto si ripassa come mem0
    m2 = modello(20, tempo=4, seed=1).eval()
    with torch.no_grad():
        z, h = m2(ids, pad, causal=True,
                  mem0=m.mem)
    assert torch.isfinite(h).all()


def test_tempo_con_padding_finito():
    """Blocchi fatti solo di padding non devono produrre NaN."""
    for training in (False, True):
        m = modello(20, tempo=4)
        m.train(training)
        ids, pad = _ids(2, 16, pad_da=[5, 16])
        z, h = m(ids, pad, causal=True)
        assert torch.isfinite(z).all()
        assert torch.isfinite(h).all()


def test_cresci_conserva_la_funzione():
    m = modello(20, comm=False, tempo=4).eval()
    with torch.no_grad():
        m.zinj.weight.normal_(0, 0.1)
    ids, pad = _ids(2, 12)
    z1, h1 = m(ids, pad, causal=True)
    m.cresci()
    assert m.K == 3 and len(m.upd.blocks) == 3
    z2, h2 = m.eval()(ids, pad, causal=True)
    assert z2.shape[1] == 3
    assert torch.allclose(h1, h2, atol=1e-5)
    assert torch.allclose(z1, z2[:, :2], atol=1e-5)


def test_carica_checkpoint_antico():
    m = modello(20, K=2)
    vecchio = {k: v for k, v in m.state_dict().items()
               if not k.startswith('upd.')}
    for k, v in m.upd.blocks[0].state_dict().items():
        vecchio['upd.' + k] = v.clone()
    m2 = modello(20, K=2, seed=5)
    m2.load_state_dict(vecchio)
    for b in m2.upd.blocks:
        for k, v in b.state_dict().items():
            assert torch.equal(v, vecchio['upd.' + k])


def test_param_count():
    m = MultiUniverse(50, K=1, enc_layers=1, d=32, m=4)
    assert param_count(m) > 0


def test_cresci_projected_conserva_la_funzione():
    m = modello(20, K=2, init='projected', comm=False, tempo=4).eval()
    with torch.no_grad():
        m.zinj.weight.normal_(0, 0.1)
    ids, pad = _ids(2, 12)
    z1, h1 = m(ids, pad, causal=True)
    m.cresci()
    assert m.K == 3 and m.init.K == 3 and len(m.upd.blocks) == 3
    assert m.init.W.shape[0] == 3 and m.init.b.shape[0] == 3
    z2, h2 = m.eval()(ids, pad, causal=True)
    assert z2.shape[1] == 3
    assert torch.allclose(h1, h2, atol=1e-5)
    assert torch.allclose(z1, z2[:, :2], atol=1e-5)
