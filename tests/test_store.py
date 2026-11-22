import json

import pytest

from murlen.store import Store


def _regola(st, da=9, a=1):
    return st.add('trasformazione', 'alternanza',
                  {'da': da, 'a': a, 'n': 1}, 'test',
                  ident={'da': da, 'a': a})


def test_ciclo_di_vita(tmp_path):
    st = Store(str(tmp_path / 'm.jsonl'))
    r = _regola(st)
    assert r['stato'] == 'candidata'
    st.conferma(r)
    assert st.ultima(r['id'])['stato'] == 'candidata'
    st.conferma(r)
    assert st.ultima(r['id'])['stato'] == 'consolidata'
    st.contraddice(r, 'controesempio')
    u = st.ultima(r['id'])
    assert u['stato'] == 'declassata' and u['n_contra'] == 1


def test_riosservare_conferma(tmp_path):
    st = Store(str(tmp_path / 'm.jsonl'))
    _regola(st)
    _regola(st)
    r = _regola(st)
    assert r['n_confirm'] == 2 and r['stato'] == 'consolidata'


def test_ricarico_identico(tmp_path):
    p = str(tmp_path / 'm.jsonl')
    st = Store(p)
    r = _regola(st)
    st.conferma(r)
    _regola(st, 4, 5)
    st2 = Store(p)
    assert st2.records == st.records
    assert st2.audit() == st.audit()


def test_coda_troncata(tmp_path):
    p = tmp_path / 'm.jsonl'
    st = Store(str(p))
    _regola(st)
    with open(p, 'a') as f:
        f.write('{"id": "rott')
    st2 = Store(str(p))
    assert len(st2.records) == 1
    righe = p.read_text().splitlines()
    assert len(righe) == 1 and json.loads(righe[0])


def test_revidi_e_attive(tmp_path):
    st = Store(str(tmp_path / 'm.jsonl'))
    r = _regola(st)
    r2 = st.revidi(r, 'motivo', {'da': 9, 'a': 2})
    assert r2['versione'] == 2 and r2['stato'] == 'candidata'
    att = st.attive('trasformazione')
    assert len(att) == 1 and att[0]['versione'] == 2


def test_dimentica(tmp_path):
    st = Store(str(tmp_path / 'm.jsonl'))
    for i in range(6):
        _regola(st, i, i + 1)
    assert st.dimentica('trasformazione', recenti=2) == 4
    assert len(st.attive('trasformazione')) == 2


def _osserva(st, volte):
    for _ in range(volte):
        rec = st.add('f', 'scambio', {'a': 1}, 'test')
    return rec


def test_consolida_predefinito_tre_osservazioni(tmp_path):
    st = Store(str(tmp_path / 'm.jsonl'))
    assert _osserva(st, 2)['stato'] == 'candidata'
    assert _osserva(st, 1)['stato'] == 'consolidata'


def test_m_consolida_impostabile(tmp_path):
    st = Store(str(tmp_path / 'm.jsonl'), m_consolida=4)
    assert _osserva(st, 4)['stato'] == 'candidata'
    assert _osserva(st, 1)['stato'] == 'consolidata'


def test_simili_solo_stato_corrente(tmp_path):
    """simili guarda lo stato corrente: una regola
    declassata non torna piu' attraverso le sue
    vecchie versioni."""
    st = Store(str(tmp_path / 'm.jsonl'))
    r = _regola(st)
    st.conferma(r)
    st.contraddice(r, 'controesempio')
    assert st.simili({'da': 9, 'a': 1},
                     'trasformazione') == []


def test_simili_niente_duplicati(tmp_path):
    """Piu' versioni della stessa regola non
    occupano piu' slot del top-k: una regola,
    una voce."""
    st = Store(str(tmp_path / 'm.jsonl'))
    r = _regola(st)
    st.conferma(r)
    out = st.simili({'da': 9, 'a': 1},
                    'trasformazione')
    assert len(out) == 1


def test_record_assente_errore_chiaro(tmp_path):
    """conferma/contraddice/revidi su un record mai
    visto in questo store: KeyError, non TypeError
    su None."""
    st = Store(str(tmp_path / 'm.jsonl'))
    altro = Store(str(tmp_path / 'altro.jsonl'))
    r = _regola(altro)
    for op in (lambda: st.conferma(r),
               lambda: st.contraddice(r, 'x'),
               lambda: st.revidi(r, 'x', {})):
        with pytest.raises(KeyError):
            op()


def test_m_consolida_impostabile_conferma(tmp_path):
    st = Store(str(tmp_path / 'm.jsonl'), m_consolida=3)
    rec = st.add('f', 'scambio', {'a': 1}, 'test')
    st.conferma(rec)
    st.conferma(rec)
    assert st.ultima(rec['id'])['stato'] == 'candidata'
    st.conferma(rec)
    assert st.ultima(rec['id'])['stato'] == 'consolidata'
