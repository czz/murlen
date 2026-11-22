"""store — memoria persistente su disco.
Schema mem_schema_v1.1 (contratto congelato):
JSONL append-only, mai sovrascrittura;
n_confirm/n_contra; ambiguita' preservata
(regole concorrenti tenute); ricarico
identico. Regole = candidati verificati
PRIMA dell'uso — MAI precedenza implicita.

Stati: candidata -> consolidata (>=m
conferme dopo la prima osservazione;
m = m_consolida) -> declassata
(controesempio registrato). Le regole mai
confermate e non recenti vengono
dimenticate. Ogni stato resta consultabile.
"""
import bisect
import json
import os

M_CONSOLIDA = 2  # conferme dopo la prima
                 # osservazione per consolidare


def _rid(rec):
    """ID stabile: famiglia+forma+parametri."""
    return json.dumps(
        [rec['famiglia'], rec['forma'],
         rec.get('parametri', [])],
        sort_keys=True)


def _fv(k, v):
    """Chiave d'indice per un campo dei
    parametri (stesso serializzato di _rid)."""
    return (k, json.dumps(v, sort_keys=True))


class Store:
    def __init__(self, path, m_consolida=M_CONSOLIDA):
        self.path = path
        self.m_consolida = m_consolida
        self.records = []
        self._by_id = {}
        self._inv = {}    # (campo, valore) -> {id}
        self._by_n = {}   # famiglia -> [(n_confirm, id)]
        self._f = None
        if os.path.exists(path):
            rotto = None
            with open(path) as f:
                while True:
                    pos = f.tell()
                    ln = f.readline()
                    if not ln:
                        break
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        rec = json.loads(ln)
                    except ValueError:
                        # coda troncata da un
                        # kill a meta' append:
                        # il record incompleto
                        # non e' mai esistito
                        rotto = pos
                        break
                    self._index(rec)
            if rotto is not None:
                with open(path, 'r+') as f:
                    f.truncate(rotto)

    def _index(self, rec):
        self.records.append(rec)
        rid = rec.get('id')
        if not rid:
            return
        old = self._by_id.get(rid)
        if old is not None:
            self._togli(old)
        self._by_id[rid] = rec
        self._metti(rec)

    def _metti(self, rec):
        """Indici del richiamo: campi dei
        parametri e posizione per n_confirm."""
        rid = rec['id']
        p = rec.get('parametri')
        if isinstance(p, dict):
            for k, v in p.items():
                self._inv.setdefault(
                    _fv(k, v), set()).add(rid)
        lst = self._by_n.setdefault(
            rec.get('famiglia'), [])
        bisect.insort(lst, (
            rec.get('n_confirm', 0), rid))

    def _togli(self, rec):
        """Toglie la versione precedente dagli
        indici prima di sostituirla."""
        rid = rec['id']
        p = rec.get('parametri')
        if isinstance(p, dict):
            for k, v in p.items():
                s = self._inv.get(_fv(k, v))
                if s is not None:
                    s.discard(rid)
                    if not s:
                        del self._inv[_fv(k, v)]
        lst = self._by_n.get(
            rec.get('famiglia'))
        if lst:
            pos = bisect.bisect_left(lst, (
                rec.get('n_confirm', 0), rid))
            if pos < len(lst) and lst[pos][
                    1] == rid:
                lst.pop(pos)

    def _emit(self, rec):
        self._index(rec)
        if self._f is None:
            self._f = open(self.path, 'a')
        self._f.write(json.dumps(rec) + '\n')
        self._f.flush()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __del__(self):
        self.close()

    def add(self, famiglia, forma,
            parametri, provenienza,
            tipo='regola', ident=None):
        rid = _rid({'famiglia': famiglia,
                    'forma': forma,
                    'parametri': ident
                    if ident is not None
                    else parametri})
        st = self._by_id.get(rid)
        if st is not None:
            # la stessa forma ri-osservata nel
            # mondo = episodio verificato:
            # l'evidenza si aggiorna, il
            # conteggio cresce, lo stato evolve
            n = st['n_confirm'] + 1
            stato = st['stato']
            if stato == 'candidata' and \
                    n >= self.m_consolida:
                stato = 'consolidata'
            self._emit({**st,
                        'parametri': parametri,
                        'n_confirm': n,
                        'stato': stato})
            return self.ultima(rid)
        rec = {
            'id': rid,
            'tipo': tipo,
            'famiglia': famiglia,
            'forma': forma,
            'parametri': parametri,
            'stato': 'candidata',
            'nato': len(self.records),
            'n_confirm': 0,
            'n_contra': 0,
            'versione': 1,
            'provenienza': provenienza}
        self._emit(rec)
        return rec

    def ultima(self, rid):
        """Ultimo stato di una regola
        (indice O(1))."""
        return self._by_id.get(rid)

    def _stato(self, rec):
        """Stato corrente del record. KeyError se
        il record non e' in questo store — non un
        TypeError su None."""
        st = self.ultima(rec['id'])
        if st is None:
            raise KeyError(rec['id'])
        return st

    def simili(self, parametri, famiglia,
               k=3):
        """Richiamo per similarita': i record
        con piu' campi uguali a quelli dati.
        Ogni regola conta con il suo stato
        corrente: le versioni vecchie restano
        storia, non candidati (una declassata
        non torna, niente duplicati)."""
        def punti(r):
            p = r['parametri']
            # attenzione: i match contano, il
            # rinforzo hebbiano premia le
            # memorie gia' richiamate
            return sum(1 for kk, vv in
                       parametri.items()
                       if p.get(kk) == vv) + \
                r.get('n_confirm', 0) * 0.25
        # candidati dall'indice invertito:
        # chi condivide almeno un campo, piu'
        # i top-k per n_confirm nella famiglia
        # (possono vincere anche a 0 match)
        ids = set()
        for kk, vv in parametri.items():
            ids |= self._inv.get(_fv(kk, vv),
                                 set())
        for n, rid in self._by_n.get(
                famiglia, [])[-k:]:
            ids.add(rid)
        cand = [r for rid in ids
                if (r := self._by_id.get(rid))
                is not None
                and r.get('famiglia') ==
                famiglia
                and r['stato'] not in
                ('declassata', 'dimenticata')]
        return sorted(cand, key=punti,
                      reverse=True)[:k]

    def conferma(self, rec):
        """Episodio verificato: +1; a
        m_consolida conferme -> consolidata."""
        st = self._stato(rec)
        stato = st['stato']
        n = st['n_confirm'] + 1
        if stato == 'candidata' and \
                n >= self.m_consolida:
            stato = 'consolidata'
        self._emit({**st,
                    'n_confirm': n,
                    'stato': stato})

    def contraddice(self, rec, prova):
        """Controesempio: regola fuori dal
        fast-path; prova registrata."""
        st = self._stato(rec)
        self._emit({**st,
                    'n_contra':
                        st['n_contra'] + 1,
                    'stato': 'declassata',
                    'ultima_prova_contra':
                        prova})

    def revidi(self, rec, motivo,
               nuovi_parametri):
        """Nuova versione: vecchia
        consultabile, mai persa."""
        st = self._stato(rec)
        self._emit({**st,
                    'parametri':
                        nuovi_parametri,
                    'versione':
                        st['versione'] + 1,
                    'stato': 'candidata',
                    'n_confirm': 0,
                    'motivo': motivo})
        return self.ultima(rec['id'])

    def attive(self, famiglia=None):
        """Regole non declassate, stato
        corrente. Senza famiglia solo i
        record tipo='regola'; con famiglia
        tutti i tipi — serve a dimentica
        per l'oblio anche degli episodi."""
        out = []
        for r in self._by_id.values():
            if famiglia is None:
                if r.get('tipo') != 'regola':
                    continue
            elif r.get('famiglia') != famiglia:
                continue
            if r['stato'] not in (
                    'declassata', 'dimenticata'):
                out.append(r)
        return out

    def dimentica(self, famiglia,
                  recenti=500):
        """Oblio: gli episodi mai rinforzati
        e non recenti decadono."""
        cand = self.attive(famiglia)
        if len(cand) <= recenti:
            return 0
        d = 0
        for r in cand[:len(cand) - recenti]:
            if r['n_confirm'] == 0:
                self._emit({**r,
                            'stato': 'dimenticata'})
                d += 1
        return d

    def consolidate(self):
        return [r for r in self.attive()
                if r['stato'] ==
                'consolidata']

    def audit(self):
        return {
            'n_records': len(self.records),
            'attive': len(self.attive()),
            'consolidate':
                len(self.consolidate()),
            'declassate':
                sum(1 for r in
                    self._by_id.values()
                    if r.get('tipo') ==
                    'regola' and
                    r['stato'] ==
                    'declassata')}
