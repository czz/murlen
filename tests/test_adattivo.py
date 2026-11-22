"""Batch adattivo: a ogni passo il batch riempie la VRAM libera e
segue gli altri programmi (sale e scende). Se la memoria viene
rifiutata a meta' passo, il passo si ripete con meno sequenze."""
import torch
import pytest

from murlen import ponte
from murlen.model import MultiUniverse
from murlen.ponte import (vocab_da_sequenze, impara_predizione,
                          BatchAdattivo, tensorizza)
from aiuti import (modello, sequenze_variabili,
                   sequenze_ripetitive)


def _conta_batch(core, rifiuta=lambda k, nb: False):
    """Registra i batch riusciti; `rifiuta(tentativo, nb)` simula
    la memoria rifiutata."""
    visti, tentativi = [], []
    orig = core.enc.forward

    def fwd(X, *a, **kw):
        k = len(tentativi)
        tentativi.append(X.size(0))
        if rifiuta(k, X.size(0)):
            raise torch.OutOfMemoryError('simulato')
        out = orig(X, *a, **kw)
        visti.append(X.size(0))
        return out
    core.enc.forward = fwd
    return visti, tentativi


def _uguali(B=16):
    return [[3, 5, 7, 9] * 10 for _ in range(B)]


def test_senza_gpu_batch_fisso():
    seqs = sequenze_variabili()
    vocab, n = vocab_da_sequenze(seqs)
    core = modello(n)
    visti, _ = _conta_batch(core)
    impara_predizione(core, seqs, vocab, passi=3)
    assert visti == [4, 4, 4]


def test_prima_misura():
    ad = BatchAdattivo(16, 100)
    assert ad.scegli(None) == 16          # senza GPU: il massimo
    assert ad.scegli(10**9) == 1          # nessuna misura: una sola
    ad.registra(1, 100, 1100)
    assert ad.scegli(11500) == 10         # stima dalla prima misura
    assert ad.scegli(0) == 1              # mai sotto 1
    assert ad.scegli(10**9) == 16         # mai oltre il massimo


def test_picco_nullo_non_divide():
    """Misure degeneri (picco 0): niente divisione
    per zero, si torna al massimo."""
    ad = BatchAdattivo(16, 100)
    ad.registra(1, 100, 0)
    assert ad.scegli(10**9) == 16


def test_stima_precisa():
    """Memoria vera = 100 fissi + 1000 a sequenza: dopo poche
    misure la stima e' esatta e la VRAM si riempie tutta."""
    ad = BatchAdattivo(32, 100)
    for nb in (1, 4, 8):
        ad.registra(nb, 100, 100 + nb * 1000)
    assert ad.scegli(16500) == 16
    assert ad.scegli(8500) == 8
    assert ad.scegli(4900) == 4


def test_stima_con_lunghezze_diverse():
    """Batch di lunghezze diverse: la stima vale per la lunghezza
    massima (la memoria ha anche un termine quadratico)."""
    ad = BatchAdattivo(32, 100)

    def mem(nb, l):
        t = l / 100
        return 50 + nb * (600 * t + 400 * t * t)
    for nb, l in ((1, 100), (2, 50), (4, 100), (3, 80), (6, 30)):
        ad.registra(nb, l, mem(nb, l))
    assert ad.scegli(10500) == 10


def _vram_finta(monkeypatch, visti, libera):
    monkeypatch.setattr(ponte, '_vram', libera)
    monkeypatch.setattr(ponte, '_azzera_picco', lambda dev: 0)
    monkeypatch.setattr(ponte, '_picco',
                        lambda dev, base: 100 + visti[-1] * 1000)


def test_vigilanza_fa_crescere():
    """Se la loss resta oltre soglia, cresci() scatta da
    solo fino a universi_max. Senza vigilanza o con soglia
    irraggiungibile, niente nascite."""
    rnd = torch.Generator().manual_seed(1)
    seqs = [torch.randint(2, 40, (30,),
            generator=rnd).tolist() for _ in range(8)]
    vocab, n = vocab_da_sequenze(seqs)

    def gira(**kw):
        torch.manual_seed(0)
        core = modello(n, K=1)
        impara_predizione(core, seqs, vocab, passi=30,
                          riscaldamento=0, **kw)
        return core

    assert gira().K == 1                       # mai attivata
    assert gira(vigilanza=99., pazienza=5).K == 1   # irraggiungibile
    c = gira(vigilanza=0.5, pazienza=5, universi_max=3)
    assert c.K == 3                            # scatta e si ferma al cap
    assert len(c.upd.blocks) == 3
    assert c.gate.shape[0] == 3


def test_vigilanza_ignora_le_esplosioni(monkeypatch):
    """Passi con loss non finita (grad non finito ->
    step saltato) non contano per la pazienza:
    cresci() non scatta su un'esplosione."""
    seqs = _uguali()
    vocab, n = vocab_da_sequenze(seqs)
    torch.manual_seed(0)
    core = modello(n, K=1)
    perdite = iter([float('inf')] * 10)
    monkeypatch.setattr(
        ponte, '_passo', lambda *a: next(perdite))
    impara_predizione(core, seqs, vocab, passi=10,
                      riscaldamento=0, vigilanza=0.5,
                      pazienza=3, universi_max=3)
    assert core.K == 1


def _repulsione(core, seqs, vocab):
    """Quanto le voci si somigliano (fuori diagonale
    della Gram dei riassunti sullo stato finale)."""
    ids, pad = tensorizza(seqs, vocab, 64)
    with torch.no_grad():
        z, _ = core(ids, pad)
    s = torch.stack([core.comm.pool(z[:, k])
                     for k in range(core.K)], 1)
    sn = torch.nn.functional.normalize(s, dim=-1)
    g = sn @ sn.transpose(1, 2) - torch.eye(core.K)
    return float(g.square().mean())


def test_diversita_allontana_le_voci():
    """Con la repulsione attiva le voci si somigliano
    meno che senza."""
    seqs = sequenze_ripetitive()
    vocab, n = vocab_da_sequenze(seqs)

    def rep(diversita):
        core = modello(n, tempo=8, seed=3)
        impara_predizione(core, seqs, vocab, passi=15,
                          diversita=diversita)
        return _repulsione(core, seqs, vocab)

    assert rep(10.) < rep(0.)


def test_impara_segue_la_vram(monkeypatch):
    """VRAM simulata (100 fissi + 1000 a sequenza): abbondante,
    poi un altro programma ne prende, poi la libera."""
    seqs = _uguali()
    vocab, n = vocab_da_sequenze(seqs)
    core = modello(n)
    visti, _ = _conta_batch(core)
    libera = iter([10**9, 8100, 8100, 8200, 3500, 3500, 3500,
                   16600, 16600, 16600])
    _vram_finta(monkeypatch, visti, lambda dev: next(libera))
    hist, _ = impara_predizione(core, seqs, vocab, passi=10)
    assert visti == [1, 7, 7, 8, 3, 3, 3, 16, 16, 16]
    assert len(hist) == 10


def test_memoria_presa_a_meta_passo(monkeypatch):
    """Un altro programma prende VRAM mentre il passo e' in
    corso: il passo si ripete con quello che ci sta, nessun
    errore, nessun passo perso; poi il batch risale."""
    seqs = _uguali()
    vocab, n = vocab_da_sequenze(seqs)
    core = modello(n)
    stato = {'occupato': False}

    def rifiuta(k, nb):
        if k == 5:
            stato['occupato'] = True
        if k == 9:
            stato['occupato'] = False
        return stato['occupato'] and nb > 3
    visti, tentativi = _conta_batch(core, rifiuta)
    _vram_finta(monkeypatch, visti,
                lambda dev: 3500 if stato['occupato'] else 16600)
    hist, _ = impara_predizione(core, seqs, vocab, passi=10)
    assert len(hist) == 10
    assert visti == [1, 15, 16, 16, 16, 3, 3, 3, 3, 16]
    assert tentativi[5] == 16                 # rifiutato e ripetuto


def test_rifiutata_senza_gpu_riprova_con_meno():
    seqs = sequenze_variabili()
    vocab, n = vocab_da_sequenze(seqs)
    core = modello(n)
    visti, tentativi = _conta_batch(core, lambda k, nb: k < 2)
    hist, _ = impara_predizione(core, seqs, vocab, passi=3)
    assert tentativi[:3] == [4, 2, 1]
    assert visti == [1, 4, 4] and len(hist) == 3


def test_nemmeno_una_sequenza_aspetta(monkeypatch):
    """Nessuna VRAM nemmeno per una sequenza: aspetta che si
    liberi e riprova, invece di fermarsi."""
    seqs = sequenze_variabili()
    vocab, n = vocab_da_sequenze(seqs)
    core = modello(n)
    attese = []
    monkeypatch.setattr(ponte.time, 'sleep', attese.append)
    visti, _ = _conta_batch(core, lambda k, nb: k < 3)
    hist, _ = impara_predizione(core, seqs, vocab, passi=2,
                                batch=1)
    assert len(attese) == 3 and visti == [1, 1]


cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason='CUDA non disponibile')


def _lunghe(B=16):
    g = torch.Generator().manual_seed(0)
    return [torch.randint(0, 200, (1024,), generator=g).tolist()
            for _ in range(B)]


@cuda
def test_gpu_vera_sequenze_lunghe():
    """1024 simboli, 3 universi, tempo=64, tutte insieme: la GPU
    da 4 GB non le regge, il batch deve adattarsi da solo."""
    seqs = _lunghe(8)
    vocab, n = vocab_da_sequenze(seqs)
    torch.manual_seed(0)
    core = MultiUniverse(n, K=3, comm=True, tempo=64)
    hist, _ = impara_predizione(core, seqs, vocab, passi=4,
                                device='cuda')
    assert len(hist) == 4 and all(h == h for h in hist)


@cuda
def test_gpu_vera_segue_altri_programmi():
    """Un 'altro programma' prende il 40% della VRAM a meta'
    training (anche a passo in corso) e poi la libera: il batch
    scende e poi risale, senza errori."""
    seqs = _lunghe()
    vocab, n = vocab_da_sequenze(seqs)
    torch.manual_seed(0)
    core = MultiUniverse(n, K=3, comm=True, tempo=0)
    visti, occupa = [], []
    orig = core.enc.forward
    torch.cuda.empty_cache()   # i test precedenti lasciano cache
    libera0 = torch.cuda.mem_get_info()[0]

    def fwd(X, *a, **kw):
        if len(visti) == 10 and not occupa:
            occupa.append(torch.empty(int(libera0 * 0.4),
                                      dtype=torch.uint8,
                                      device='cuda'))
        if len(visti) == 20:
            occupa.clear()
        out = orig(X, *a, **kw)
        visti.append(X.size(0))
        return out
    core.enc.forward = fwd
    hist, _ = impara_predizione(core, seqs, vocab, passi=30,
                                device='cuda')
    print('batch per passo:', visti)
    assert len(hist) == 30
    prima, durante, dopo = visti[7:10], visti[12:20], visti[25:]
    assert max(durante) < min(prima)
    assert min(dopo) > max(durante)
