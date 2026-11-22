"""Test su GPU: saltati se CUDA non e' disponibile."""
import pytest
import torch

from murlen.ponte import vocab_da_sequenze, impara_predizione
from aiuti import modello, sequenze_ripetitive

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason='CUDA non disponibile')


@cuda
def test_cresci_su_gpu():
    m = modello(20, comm=True, tempo=4).cuda()
    m.cresci()
    for nome, p in m.named_parameters():
        assert p.is_cuda, nome
    ids = torch.randint(2, 20, (2, 12), device='cuda')
    pad = torch.zeros(2, 12, dtype=torch.bool, device='cuda')
    z, h = m(ids, pad, causal=True)
    assert z.shape[1] == 3 and torch.isfinite(h).all()


@cuda
@pytest.mark.parametrize('tempo,batch', [(0, None), (0, 2), (8, None)])
def test_impara_su_gpu(tempo, batch):
    seqs = sequenze_ripetitive()
    vocab, n = vocab_da_sequenze(seqs)
    core = modello(n, tempo=tempo)
    hist, testa = impara_predizione(core, seqs, vocab, passi=30,
                                    device='cuda', batch=batch)
    assert all(map(lambda x: x == x, hist))
    assert hist[-1] < hist[0]
    assert next(testa.parameters()).is_cuda


@cuda
def test_gpu_come_cpu():
    """Stesso primo passo su CPU e GPU (a meno di arrotondamenti)."""
    seqs = sequenze_ripetitive()
    vocab, n = vocab_da_sequenze(seqs)
    out = []
    for dev in ('cpu', 'cuda'):
        core = modello(n)
        hist, _ = impara_predizione(core, seqs, vocab, passi=1,
                                    device=dev, batch=1)
        out.append(hist[0])
    assert abs(out[0] - out[1]) < 1e-3


@cuda
def test_cresci_da_uno_su_gpu():
    m = modello(20, K=1, comm=None, tempo=None).cuda()
    m.cresci()
    assert m.comm_on
    for nome, p in m.named_parameters():
        assert p.is_cuda, nome
