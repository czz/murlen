"""gateway — la porta fra il foglietto e l'archivio.

Message broker bidirezionale fra le tracce del
foglietto e la memoria persistente (Store):

- `veglia`: una finestra vissuta che lascia tracce
  nel foglietto (oltre al confine d'episodio, che
  e' sempre tracciato) viene annotata come
  episodio del giorno — simboli grezzi, nessuna
  interpretazione;
- `sonno`: gli episodi del giorno entrano
  nell'archivio (Store), l'oblio corre su quelli
  mai rinforzati, e gli episodi attivi tornano
  come materiale di replay per il consolidamento.

Le tracce del foglietto sono vettori h senza
simboli: l'episodio archiviabile e' la finestra
che le ha prodotte, catturata al confine. La
scelta di cosa merita memoria resta del
foglietto (il suo fence dei quartili) — qui si
chiede solo "questa finestra ha scritto?".
Copre il percorso a fence; la variante a gate
appreso (mem_gate) non e' ancora agganciata.

Il buffer del giorno e' in RAM: un processo che
muore prima del sonno perde gli episodi non
ancora consegnati all'archivio.
"""


class Gateway:
    """Gateway <-> archivio (Store).

    `veglia` raccoglie gli episodi della giornata;
    `sonno` li consegna all'archivio e restituisce
    il materiale di replay."""

    def __init__(self, store):
        self.store = store
        self._giorno = []   # episodi vissuti, non
                            # ancora consegnati

    def veglia(self, ids, sorpresa, pad, foglio):
        """Annota le finestre vissute (B,N) che
        hanno lasciato tracce oltre al confine:
        per ogni elemento del batch la scelta e'
        quella del foglietto stesso. -> numero di
        episodi annotati."""
        s = sorpresa[:, 1:]
        p = pad[:, 1:]
        if not s.numel():
            return 0
        sel = foglio.scegli_valori(s, p)
        n0 = 0
        for b in range(ids.size(0)):
            if not bool(sel[b].any()):
                continue
            n = int((~pad[b]).sum())
            self._giorno.append(
                ids[b, :n].tolist())
            n0 += 1
        return n0

    def sonno(self, recenti=None):
        """Consegna gli episodi all'archivio
        (ri-osservare lo stesso episodio nel
        mondo = conferma vera, lo decide il
        Store per identita' dei simboli), fa
        decadere i mai-rinforzati e restituisce
        il materiale di replay: gli episodi
        attivi in ordine di nascita. `recenti`
        passa il limite di ritenzione al
        dimentica dello Store."""
        for ep in self._giorno:
            self.store.add('episodio', 'vissuto',
                           {'simboli': ep},
                           'gateway',
                           tipo='episodio')
        self._giorno = []
        self.store.dimentica(
            'episodio',
            **({'recenti': recenti}
               if recenti is not None else {}))
        att = self.store.attive('episodio')
        att.sort(key=lambda r: r['nato'])
        return [r['parametri']['simboli']
                for r in att]
