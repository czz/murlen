"""ponte — i simboli diventano i token di
MultiUniverse.

Le sequenze di id-simbolo, rimappate su un
vocabolario contiguo, entrano nell'encoder:
il modello impara a predire il simbolo
successivo — auto-supervisione, il segnale
viene dai dati, non da etichette.

Il dizionario simbolo->id NON e' scritto: gli
id in ingresso sono conservati, solo
compattati in un range.
"""
import math
import os
import time
from collections import deque

import torch
import torch.nn as nn

PAD_ID = 1


def vocab_da_sequenze(seqs):
    """simboli visti -> vocabolario contiguo
    (0 = EOS, 1 = PAD riservati). Gli id in
    ingresso sono conservati, solo compressi
    nel range dell'embedding."""
    gids = sorted({g for s_ in seqs for g in s_})
    return {g: i + 2 for i, g in enumerate(gids)}, \
        len(gids) + 2


def tensorizza(seqs, vocab, ctx=1024):
    """sequenze di simboli -> tensori di id
    (padding a destra)."""
    out, pad = [], []
    for s_ in seqs:
        ids = [vocab[g] for g in s_][:ctx]
        p = [False] * len(ids) + \
            [True] * (ctx - len(ids))
        ids += [PAD_ID] * (ctx - len(ids))
        out.append(ids)
        pad.append(p)
    return (torch.tensor(out), torch.tensor(pad))


class TestaPredizione(nn.Module):
    """h -> distribuzione sul prossimo token.
    Predittore generico — nessuna conoscenza
    di cosa rappresentino i token."""

    def __init__(self, vocab, d):
        super().__init__()
        self.lin = nn.Linear(d, vocab)

    def forward(self, h):
        return self.lin(h)


def _flusso(core, X, pad, scrivi=None):
    """h per la predizione. Senza tempo (o con
    tempo >= N) h e' l'uscita dell'encoder e gli
    universi non la toccano: non si calcolano."""
    if not core.tempo or core.tempo >= X.size(1):
        return core.enc(X, pad, causal=True)
    return core(X, pad, causal=True, scrivi=scrivi)[1]


def _scrivi_sorpresa(testa, ids):
    """Callback `scrivi` per forward: sorpresa per
    posizione del blocco che inizia ad `a` — quello
    che annota() usa per decidere cosa lascia
    traccia. Posizione 0 = inf (confine di
    blocco). no_grad: alla scelta basta il valore,
    il gradiente va al lato lettura."""
    def f(a, h_j, p_j):
        with torch.no_grad():
            lo = testa(h_j)
            nj = h_j.size(1)
            s = h_j.new_full((h_j.size(0), 1),
                             float('inf'))
            ce = nn.functional.cross_entropy(
                lo[:, :-1].reshape(-1, lo.size(-1)),
                ids[:, a + 1:a + nj].reshape(-1),
                reduction='none',
                ignore_index=PAD_ID).view(
                    h_j.size(0), -1)
            return torch.cat([s, ce], 1)
    return f


MARGINE = 0.05     # quota di VRAM lasciata libera
ATTESA_MAX = 300   # secondi di attesa se la VRAM
                   # non basta nemmeno per una
                   # sequenza


def _vram(device):
    """Byte di VRAM disponibili ora (None se non
    CUDA): liberi sulla scheda + cache di PyTorch
    non usata, meno il margine. Tiene conto
    degli altri programmi aperti."""
    device = torch.device(device)
    if device.type != 'cuda':
        return None
    libera, totale = torch.cuda.mem_get_info(device)
    cache = (torch.cuda.memory_reserved(device) -
             torch.cuda.memory_allocated(device))
    return max(0, libera + cache - MARGINE * totale)


def _azzera_picco(device):
    if torch.device(device).type != 'cuda':
        return None
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_allocated(device)


def _picco(device, base):
    if base is None:
        return None
    return torch.cuda.max_memory_allocated(device) - base


def _svuota(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()


class BatchAdattivo:
    """Quante sequenze per passo per riempire la
    VRAM libera. Ogni passo misura la memoria
    usata; dalle misure si ricava
    memoria = fisso + nb * (b*t + c*t^2),
    t = lunghezza / L, e quindi il costo di una
    sequenza lunga L. Il primo passo (una sola
    sequenza) include lo stato dell'ottimizzatore
    e serve solo finche' mancano altre misure."""
    CALO = 0.95

    def __init__(self, massimo, L):
        self.massimo, self.L = massimo, L
        self.prima = None
        self.misure = []
        self.correzione = 1.

    def scegli(self, libera):
        if libera is None:
            return self.massimo
        if self.prima is None:
            return 1
        fisso, seq = self._stima()
        if seq <= 0:
            # misure degeneri (picchi nulli):
            # niente da stimare, si prende il
            # massimo e l'eventuale rifiuto
            # correggera'
            return self.massimo
        k = self.correzione
        nb = int((libera - fisso * k) // (seq * k))
        return max(1, min(self.massimo, nb))

    def registra(self, nb, l, picco):
        m = (nb, l / self.L, float(picco))
        if self.prima is None:
            self.prima = m
        else:
            self.misure = (self.misure + [m])[-20:]
        self.correzione = max(
            1., self.correzione * self.CALO)

    def rifiuto(self):
        """Memoria rifiutata senza che altri ne
        abbiano presa: la stima era ottimista."""
        self.correzione *= 1.25

    def _stima(self):
        ms = self.misure or [self.prima]
        prudente = (0., max(p / (nb * t * t)
                            for nb, t, p in ms))
        if len(self.misure) < 2:
            return prudente
        A = torch.tensor([[1., nb * t, nb * t * t]
                          for nb, t, _ in self.misure],
                         dtype=torch.float64)
        y = torch.tensor([p for *_, p in self.misure],
                         dtype=torch.float64)
        a, b, c = (torch.linalg.pinv(A) @ y).tolist()
        if b + c <= 0:
            return prudente
        return max(a, 0.), b + c


NORME_STORIA = 200        # capacita' dichiarata: quante
                          # norme di gradiente il recinto
                          # del clip ricorda
NORME_MIN = 8             # sotto: statistica troppo corta,
                          # niente clip


def _recinto_norme(norme):
    """Il soffitto del clip dalla distribuzione delle norme
    di gradiente osservate: il fence dei quartili (Q3 + 1.5
    IQR) — nessun numero scelto a mano, lo decide il
    processo. None se la storia e' troppo corta per una
    statistica."""
    if norme is None or len(norme) < NORME_MIN:
        return None
    s = sorted(norme)
    q1, q3 = s[len(s) // 4], s[3 * len(s) // 4]
    return q3 + 1.5 * (q3 - q1)


def _passo(core, testa, opt, ib, pb, device, diversita=0.,
           norme=None):
    """Un passo di training -> loss."""
    ib, pb = ib.to(device), pb.to(device)
    X, Y = ib[:, :-1], ib[:, 1:]
    padi = pb[:, :-1]
    con_z = diversita and core.comm is not None \
        and core.tempo and core.tempo < X.size(1)
    # il foglietto riceve gradiente solo se qualcuno
    # scrive tracce: senza tempo non esiste, con
    # tempo>=N non c'e' chi le legge. Con scrittura
    # appresa il gate consolida da solo, senza
    # bisogno della sorpresa
    scrivi = _scrivi_sorpresa(testa, ib) if (
        core.mem_on and core.tempo
        and core.tempo < X.size(1)
        and core.foglio.scrittura is None) else None
    if con_z:
        z, h = core(X, padi, causal=True, scrivi=scrivi)
    else:
        h = _flusso(core, X, padi, scrivi)
    logits = testa(h)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        Y.reshape(-1),
        ignore_index=PAD_ID)
    if con_z:
        # repulsione fra voci: le componenti fuori
        # diagonale della Gram dei riassunti pagano —
        # le voci si spingono verso l'ortogonalita',
        # cosi' parlarsi vuol dire dire cose diverse
        s = torch.stack([core.comm.pool(z[:, k])
                         for k in range(core.K)], 1)
        sn = nn.functional.normalize(s, dim=-1)
        g = sn @ sn.transpose(1, 2)
        g = g - torch.eye(core.K, device=device)
        loss = loss + diversita * g.square().mean()
    opt.zero_grad()
    loss.backward()
    par = list(core.parameters()) + list(testa.parameters())
    tetto = _recinto_norme(norme)
    ng = nn.utils.clip_grad_norm_(
        par, tetto if tetto is not None else float('inf'))
    if torch.isfinite(ng):
        if norme is not None:
            norme.append(float(ng))
        opt.step()
    else:
        opt.zero_grad()
    return loss.item()


def impara_predizione(core, seqs, vocab,
                      passi=50, ctx=None,
                      device='cpu', lr=3e-4,
                      batch=None, seed=0,
                      riscaldamento=None, testa=None,
                      vigilanza=None, pazienza=20,
                      universi_max=8, diversita=0.):
    """Auto-supervisione sulle sequenze:
    ad ogni posizione, predire il token che
    viene dopo. -> storico della loss.
    `ctx`: simboli per sequenza (None = il
    contesto del modello).
    `testa`: testa di predizione di una
    chiamata precedente, per continuare ad
    allenarla (None = nuova).
    `riscaldamento`: passi in cui lr sale da
    lr/n a lr (None = passi//5, max 200;
    0 = lr fisso): evita che i primi passi,
    su pesi casuali, blocchino l'addestramento.
    `batch`: sequenze per passo al massimo
    (None = tutte); l'ordine e' rimescolato a
    ogni giro con `seed`, senza toccare il
    generatore globale.
    `vigilanza`: soglia di sorpresa assoluta (None =
    crescita solo a mano). Se la loss resta sopra la
    soglia per `pazienza` passi di fila, gli universi
    che ci sono non spiegano: cresci() scatta da solo e
    l'ottimizzatore viene ricreato per includere il
    nuovo (stesso lr corrente). `universi_max` ferma la
    crescita. La sorpresa comanda la nascita, come la
    vigilanza dell'ART: cio' che non si spiega diventa
    un universo in piu'.
    Il clip del gradiente non e' un numero scelto: il
    soffitto esce dalla distribuzione delle norme osservate
    (fence dei quartili, storia `NORME_STORIA`) — si adatta
    da solo alla scala del compito.
    `diversita`: peso della repulsione fra le voci
    (0 = spenta). Se >0, le voci troppo simili fra loro
    pagano in loss: pressione perche' ogni universo dica
    qualcosa di diverso — parlarsi conviene solo se ci
    si dice cose diverse.
    Su GPU il batch riempie a ogni passo la VRAM
    libera in quel momento: scende se altri
    programmi la occupano, risale quando la
    liberano. Se la memoria viene rifiutata a
    passo in corso, il passo si ripete con le
    sequenze che ci stanno (con una sola, si
    aspetta che la VRAM si liberi).
    Con `tempo` attivo e `mem` acceso il
    foglietto si consolida a ogni blocco del
    flusso (sorpresa dalla testa, o gate appreso
    con `mem_gate`): l'organo episodico riceve
    gradiente dentro la finestra, non solo fra
    finestre."""
    massimo = core.enc.pos.num_embeddings
    ctx = massimo if ctx is None else ctx
    if ctx > massimo:
        raise ValueError('ctx=%d supera il contesto del modello (%d)'
                         % (ctx, massimo))
    core.to(device)
    nv = max(vocab.values()) + 1
    if testa is None:
        testa = TestaPredizione(
            nv, core.enc.tok.embedding_dim)
    elif testa.lin.out_features < nv:
        raise ValueError('testa per %d simboli, il vocabolario '
                         'ne ha %d' % (testa.lin.out_features, nv))
    testa.to(device)
    # il corpus resta una lista di simboli: si
    # tensorizza solo il batch scelto, alla
    # larghezza della sua sequenza piu' lunga —
    # niente tensore B x ctx materializzato
    # una sequenza di 0-1 simboli non ha nulla da
    # predire (niente X): non entra nel corpus
    seqs = [s for s in list(seqs) if len(s) > 1]
    if not seqs:
        raise ValueError('nessuna sequenza con almeno '
                         '2 simboli: niente da predire')
    B = len(seqs)
    if batch is None or batch >= B:
        batch = B
    L = min(ctx, max((len(s) for s in seqs),
                     default=0))
    ad = BatchAdattivo(batch, L)
    gen = torch.Generator().manual_seed(seed)
    ordine = []
    opt = torch.optim.AdamW(
        list(core.parameters()) +
        list(testa.parameters()), lr=lr)
    w = min(200, passi // 5) if riscaldamento is None \
        else riscaldamento
    w = max(1, w)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1., (i + 1) / w))
    hist, sopra = [], 0
    # il clip non e' un numero scelto: il recinto dei
    # quartili sulle norme di gradiente osservate decide
    # il soffitto, aggiornato a ogni passo
    norme = deque(maxlen=NORME_STORIA)
    for _ in range(passi):
        libera = _vram(device)
        nb = ad.scegli(libera)
        attesa = 0
        while True:
            if nb >= B:
                sel = torch.arange(B)
            else:
                if len(ordine) < nb:
                    ordine = torch.randperm(
                        B, generator=gen).tolist()
                sel = torch.tensor(ordine[:nb])
                ordine = ordine[nb:]
            sel = sel.tolist()
            ib, pb = tensorizza(
                (seqs[i] for i in sel), vocab,
                min(ctx, max(len(seqs[i])
                             for i in sel)))
            base = _azzera_picco(device)
            try:
                hist.append(_passo(core, testa, opt,
                                   ib, pb, device, diversita,
                                   norme))
                break
            except torch.OutOfMemoryError:
                pass
            # memoria rifiutata: si libera e si
            # ripete il passo con meno sequenze
            opt.zero_grad()
            _svuota(device)
            if nb < B:
                ordine = sel + ordine
            dopo = _vram(device)
            if nb == 1:
                if attesa >= ATTESA_MAX:
                    raise MemoryError(
                        'VRAM insufficiente anche per '
                        'una sequenza da %d s' % attesa)
                time.sleep(1)
                attesa += 1
            else:
                if dopo is None or libera is None \
                        or dopo >= libera:
                    ad.rifiuto()
                nb = min(ad.scegli(dopo), nb // 2)
            libera = dopo
        sched.step()
        picco = _picco(device, base)
        if picco is not None:
            ad.registra(nb, ib.size(1), picco)
        # vigilanza: sorpresa persistente oltre soglia ->
        # nasce un universo (l'optimizer va ricreato).
        # Una loss non finita e' un'esplosione
        # transitoria (il passo e' stato saltato),
        # non una misura: non conta per la pazienza
        if vigilanza is not None and core.K < universi_max:
            if math.isfinite(hist[-1]):
                sopra = sopra + 1 if hist[-1] > vigilanza \
                    else 0
            if sopra >= pazienza:
                core.cresci()
                opt = torch.optim.AdamW(
                    list(core.parameters()) +
                    list(testa.parameters()),
                    lr=sched.get_last_lr()[0])
                sched = torch.optim.lr_scheduler.LambdaLR(
                    opt, lambda i: 1.)
                sopra = 0
    return hist, testa


def _tensori(x):
    """Itera i tensori dentro un dict/lista
    annidati (state_dict, tracce del
    foglietto)."""
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _tensori(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _tensori(v)


def _finiti(sd):
    """True se ogni tensore e' finito — un
    checkpoint NaN non si scrive e non si
    carica mai."""
    return all(torch.isfinite(t).all()
               for t in _tensori(sd))


def salva_sicuro(core, path, testa=None,
                 z=None, mem=None):
    """Checkpoint con garanzie: ogni tensore
    dev'essere finito (mai NaN su disco), la
    scrittura e' atomica (.tmp + replace, mai
    un file a meta') e il file precedente
    diventa .prev — l'ultimo sano resta
    sempre. z/mem vanno staccati dal grafo.
    -> True se scritto, False se rifiutato."""
    sd = {'modello': core.state_dict()}
    if testa is not None:
        sd['testa'] = testa.state_dict()
    if z is not None:
        sd['z'] = z.detach().cpu()
    if mem is not None:
        sd['mem'] = [t.detach().cpu()
                     for t in mem]
    if not _finiti(sd):
        return False
    tmp = path + '.tmp'
    torch.save(sd, tmp)
    if os.path.exists(path):
        os.replace(path, path + '.prev')
    os.replace(tmp, path)
    return True


def carica_sicuro(core, path, testa=None,
                  map_location=None):
    """Carica il checkpoint piu' recente sano:
    file corrotto, troncato o non finito ->
    ripiego su .prev.
    -> il dict salvato (con z/mem se c'erano),
    None se non c'e' niente di sano."""
    for p in (path, path + '.prev'):
        if not os.path.exists(p):
            continue
        try:
            sd = torch.load(
                p, weights_only=True,
                map_location=map_location)
        except Exception:
            continue
        if _finiti(sd) and 'modello' in sd:
            core.load_state_dict(sd['modello'])
            if testa is not None and \
                    'testa' in sd:
                testa.load_state_dict(
                    sd['testa'])
            return sd
    return None
