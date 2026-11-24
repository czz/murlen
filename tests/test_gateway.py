import torch

from murlen.store import Store
from murlen.gateway import Gateway
from aiuti import modello


def _foglio():
    return modello(20, tempo=4).eval().foglio


def _sorpresa(n, spike=()):
    s = torch.ones(1, n) * 0.1
    for p in spike:
        s[0, p] = 9.0
    return s


def _veglia(en, seq, spike=()):
    ids = torch.tensor([seq])
    pad = torch.zeros_like(ids, dtype=torch.bool)
    return en.veglia(ids, _sorpresa(len(seq), spike),
                     pad, _foglio())


def test_episodio_solo_se_scrive(tmp_path):
    en = Gateway(Store(str(tmp_path / 's.jsonl')))
    assert _veglia(en, [3, 5, 7, 9, 4, 6, 8, 2],
                   spike=(3,)) == 1
    assert _veglia(en, [3, 5, 7, 9, 4, 6, 8, 2]) == 0
    assert len(en._giorno) == 1


def test_confine_da_solo_non_basta(tmp_path):
    """Il confine d'episodio e' sempre tracciato:
    non conta come motivo per archiviare."""
    en = Gateway(Store(str(tmp_path / 's.jsonl')))
    assert _veglia(en, [3, 5, 7, 9, 4, 6, 8, 2],
                   spike=(0,)) == 0


def test_sonno_archivia_e_replay(tmp_path):
    st = Store(str(tmp_path / 's.jsonl'))
    en = Gateway(st)
    seqs = [[3, 5, 7, 2, 4, 6, 8, 9],
            [4, 6, 8, 3, 5, 7, 9, 2],
            [9, 8, 7, 6, 5, 4, 3, 2]]
    for q in seqs:
        _veglia(en, q, spike=(2,))
    replay = en.sonno()
    assert en._giorno == []
    recs = st.attive('episodio')
    assert len(recs) == 3 and all(
        r['tipo'] == 'episodio' for r in recs)
    assert replay == seqs          # ordine di nascita


def test_rivivere_conferma(tmp_path):
    """Lo stesso episodio vissuto di nuovo e'
    una conferma vera (identita' sui simboli):
    a m_consolida conferme -> consolidata."""
    st = Store(str(tmp_path / 's.jsonl'))
    en = Gateway(st)
    for _ in range(3):
        _veglia(en, [3, 5, 7, 2, 4, 6, 8, 9],
                spike=(2,))
        en.sonno()
    r = st.attive('episodio')[0]
    assert r['n_confirm'] == 2
    assert r['stato'] == 'consolidata'


def test_replay_esclude_dimenticati(tmp_path):
    st = Store(str(tmp_path / 's.jsonl'))
    en = Gateway(st)
    for i in range(6):
        _veglia(en, [i, 10 + i, 20 + i, 30 + i,
                     40 + i, 50 + i, 60 + i, 70 + i],
                spike=(2,))
    replay = en.sonno(recenti=2)
    attese = [[i, 10 + i, 20 + i, 30 + i,
               40 + i, 50 + i, 60 + i, 70 + i]
              for i in (4, 5)]
    assert replay == attese


def test_episodi_sopravvivono_al_ricarico(tmp_path):
    p = str(tmp_path / 's.jsonl')
    en = Gateway(Store(p))
    _veglia(en, [3, 5, 7, 2, 4, 6, 8, 9], spike=(2,))
    en.sonno()
    en.store.close()
    en2 = Gateway(Store(p))
    assert en2.sonno() == [[3, 5, 7, 2, 4, 6, 8, 9]]


def test_confine_reale_con_annota(tmp_path):
    """Stessa sorpresa per annota e per la
    porta: le tracce scritte nel foglietto e
    l'episodio archiviato coincidono."""
    m = modello(20, tempo=4).eval()
    en = Gateway(Store(str(tmp_path / 's.jsonl')))
    seq = [3, 5, 7, 9, 4, 6, 8, 2, 10, 12, 11, 13]
    ids = torch.tensor([seq])
    pad = torch.zeros_like(ids, dtype=torch.bool)
    with torch.no_grad():
        _, h = m(ids, pad, causal=True)
        sorpresa = _sorpresa(len(seq), spike=(3, 6))
        mem = m.annota(h, sorpresa, pad)
        assert en.veglia(ids, sorpresa, pad,
                         m.foglio) == 1
    assert mem is not None and mem[0].size(0) == 3
    assert en.sonno() == [seq]
