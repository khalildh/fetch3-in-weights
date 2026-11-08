"""The filesystem's meaning (the spec the compiled net is checked against; docs/tfs.md).

Token layout: the base vocabulary 0 .. V0 - 1 and S = 139 special tokens after it, in this order (the same table as
examples/wire/fs_net.tern):

    V0 + 0..4    <|fs_write|> <|fs_read|> <|fs_list|> <|fs_text|> <|fs_end|>      the model writes these
    V0 + 5..9    <|fs_mount|> <|fs_data|> <|fs_done|> <|fs_none|> <|fs_err|>      reserved: only forced (mount: runtime)
    V0 + 10..41  <|fs_new_k|>    V0 + 42..73 <|fs_ok_k|>    V0 + 74..105 <|fs_item_k|>        reserved
    V0 + 106     <|fs_delete|>                                                    the model writes it
    V0 + 107..138 <|fs_del_k|>                                                    reserved

(delete came after the first 106 tokens and is appended, so the older ids are unchanged: `upgrade` in tfs/merge.py
grows a checkpoint merged with 106 tokens by the 33 new ones.)

`Spec` reads a transcript one token at a time and says which token the filesystem forces next (or None):

    WRITE path TEXT text END    ->  NEW_k (a new file, k = the number of files so far) | OK_k (file k written: it
                                    existed, or was deleted and is re-created) | ERR (path not 1..8 tokens, or 32 files)
    READ path END               ->  DATA text DONE (the latest text written to that path) | NONE (none, or deleted) | ERR
    LIST END                    ->  ITEM_k path_k ... DONE   (the live files in index order; none: DONE)
    DELETE path END             ->  DEL_k (file k deleted) | NONE (no such live file) | ERR (path not 1..8 tokens)
    any other END               ->  ERR

A file's index belongs to its path for the whole session: deleting the file leaves the index (and its slot of the 32)
taken, and writing the path again brings the file back under the same index. Saving an image keeps only the live
files, so a mounted image starts with the indices compacted.

A forced response runs to its end (the transcript must follow it: the model's logits make it the argmax). The paths
and texts are any non-special tokens. A mounted file (MOUNT in place of TEXT, disk.py) reads its text from the image.
"""

S = 139
NF = 32
NS = 8
CALLS = ("write", "read", "list", "delete")
NAMES = ["write", "read", "list", "text", "end", "mount", "data", "done", "none", "err"]
EMIT = 5          # ids V0 + 0 .. V0 + 4 and V0 + DELETE: the model may write them; the others are reserved
DELETE = 106      # V0 + 106: <|fs_delete|>; V0 + 107 + k: <|fs_del_k|>
KIDX = ("new", "ok", "item", "del")     # the tokens that carry a file index


class Layout:
    def __init__(self, v0, nb=None):
        """nb: the net's code width when the merged vocabulary goes past this block (at least the block's own)."""
        self.v0 = v0
        self.V = v0 + S
        self.nb = max((self.V - 1).bit_length(), nb or 0)
        for i, n in enumerate(NAMES):
            setattr(self, n.upper(), v0 + i)
        self.NEW0, self.OK0, self.ITEM0 = v0 + 10, v0 + 42, v0 + 74
        self.DELETE, self.DEL0 = v0 + DELETE, v0 + DELETE + 1

    def new(self, k):
        return self.NEW0 + k

    def ok(self, k):
        return self.OK0 + k

    def item(self, k):
        return self.ITEM0 + k

    def dele(self, k):
        return self.DEL0 + k

    def emits(self):
        """The special tokens the model writes (it gets the others only when the net forces them)."""
        return list(range(self.v0, self.v0 + EMIT)) + [self.DELETE]

    def is_special(self, t):
        return t >= self.v0

    def reserved(self, t):
        return t >= self.v0 + EMIT and t != self.DELETE

    def kind(self, t):
        """'write' .. 'err', 'new', 'ok', 'item', 'delete', 'del', or None for an ordinary token."""
        if t < self.v0:
            return None
        i = t - self.v0
        if i < len(NAMES):
            return NAMES[i]
        return "new" if i < 42 else "ok" if i < 74 else "item" if i < DELETE else "delete" if i == DELETE else "del"

    def kidx(self, t):
        k = self.kind(t)
        return t - {"new": self.NEW0, "ok": self.OK0, "item": self.ITEM0, "del": self.DEL0}[k] if k in KIDX else 0

    def name(self, t):
        k = self.kind(t)
        if k is None:
            return None
        return f"<|fs_{k}_{self.kidx(t)}|>" if k in KIDX else f"<|fs_{k}|>"

    def tokens(self):
        """The special tokens' strings, in id order (for a tokenizer)."""
        return [self.name(self.v0 + i) for i in range(S)]


class Spec:
    """The filesystem state machine over a transcript. feed(t) takes the token at the next position and returns the
    token the filesystem forces at the position after it, or None. files: [[path, text]] by index (every file ever
    created, deleted ones too); live[k]: file k is not deleted; live_files(): [(k, path, text)] of the live ones."""

    def __init__(self, lay, disk=None, strict=True):
        self.lay = lay
        self.strict = strict             # False: a transcript that leaves a forced response drops the rest of it
        self.left = 0                    # forced tokens the transcript did not follow (strict=False)
        self.toks = []
        self.files = []                  # [path tuple, text tuple]
        self.index = {}                  # path -> k
        self.live = []                   # per index: not deleted
        self.disk = dict(disk or {})     # path -> text for mounted files
        self.last = None                 # latest special token: (kind, position)
        self.call = 0                    # latest call marker's position (0: BOS)
        self.text = None                 # latest text marker: (position, path, valid, mounted)
        self.queue = []                  # the rest of a forced response

    def feed(self, t):
        lay, pos = self.lay, len(self.toks)
        self.toks.append(t)
        if self.queue:
            want = self.queue.pop(0)
            if t != want and not self.strict:
                self.left += 1
                self.queue = []
            assert t == want or not self.strict, f"position {pos}: the transcript left the forced response ({t} != {want})"
        prev = self.last
        k = lay.kind(t)
        if k is not None:
            self.last = (k, pos)
        if k in CALLS:
            self.call = pos
        if k in ("text", "mount"):
            path = tuple(self.toks[self.call + 1:pos])
            ok = prev is not None and prev[0] == "write" and 1 <= len(path) <= NS
            self.text = (pos, path, ok, k == "mount")
        if k == "end":
            self.queue = self.respond(prev, pos)
        return self.queue[0] if self.queue else None

    def respond(self, prev, e):
        lay = self.lay
        pk = prev[0] if prev else None
        if pk in ("text", "mount"):
            x, path, ok, mounted = self.text
            if not ok:
                return [lay.ERR]
            text = self.disk[path] if mounted else tuple(self.toks[x + 1:e])
            if path in self.index:
                k = self.index[path]
                self.files[k][1] = text
                self.live[k] = True
                return [lay.ok(k)]
            if len(self.files) >= NF:
                return [lay.ERR]
            self.index[path] = len(self.files)
            self.files.append([path, text])
            self.live.append(True)
            return [lay.new(len(self.files) - 1)]
        if pk in ("read", "delete"):
            path = tuple(self.toks[self.call + 1:e])
            if not 1 <= len(path) <= NS:
                return [lay.ERR]
            k = self.index.get(path)
            if k is None or not self.live[k]:
                return [lay.NONE]
            if pk == "delete":
                self.live[k] = False
                return [lay.dele(k)]
            return [lay.DATA, *self.files[k][1], lay.DONE]
        if pk == "list":
            if e != self.call + 1:
                return [lay.ERR]
            out = []
            for k, p, _ in self.live_files():
                out += [lay.item(k), *p]
            return out + [lay.DONE]
        return [lay.ERR]

    def live_files(self):
        """[(k, path, text)] of the files not deleted, by index."""
        return [(k, p, t) for k, (p, t) in enumerate(self.files) if self.live[k]]


def forced(lay, toks, disk=None):
    """The forced token after each position of a transcript (None where the filesystem is silent)."""
    sp = Spec(lay, disk)
    return [sp.feed(t) for t in toks]


def session(lay, rng, n_calls=12, max_text=12, bad=0.15, paths=None, vocab=None, delete=0.15):
    """A random transcript: ordinary tokens and calls (write / read / delete / list, some malformed), every forced
    response followed. vocab: the ordinary tokens to draw from (default 1 .. V0 - 1); delete: the share of deletes
    (taken from the reads'). Returns the tokens."""
    vocab = list(vocab) if vocab is not None else list(range(1, lay.v0))
    paths = paths or [tuple(int(x) for x in rng.choice(vocab, int(rng.integers(1, 4)))) for _ in range(6)]
    sp = Spec(lay)
    toks = []

    def put(ts):
        for t in ts:
            while sp.queue:             # follow a forced response first
                toks.append(sp.queue[0])
                sp.feed(sp.queue[0])
            toks.append(t)
            nxt = sp.feed(t)
            del nxt

    def flush():
        while sp.queue:
            toks.append(sp.queue[0])
            sp.feed(sp.queue[0])

    put([0])
    for _ in range(n_calls):
        put([int(x) for x in rng.choice(vocab, int(rng.integers(0, 5)))])     # chat between calls
        r = rng.random()
        p = list(paths[int(rng.integers(len(paths)))])
        if rng.random() < bad:                   # malformed: empty / too long paths, stray markers
            p = [] if rng.random() < 0.5 else [int(x) for x in rng.choice(vocab, NS + 1 + int(rng.integers(3)))]
        if r < 0.4:
            text = [int(x) for x in rng.choice(vocab, int(rng.integers(0, max_text + 1)))]
            put([lay.WRITE, *p, lay.TEXT, *text, lay.END])
        elif r < 0.75 - delete:
            put([lay.READ, *p, lay.END])
        elif r < 0.75:
            put([lay.DELETE, *p, lay.END])
        elif r < 0.9:
            put([lay.LIST, lay.END] if rng.random() > bad else [lay.LIST, *p, lay.END])
        else:
            put([[lay.END], [lay.TEXT, lay.END], [lay.WRITE, lay.END], [lay.READ, lay.TEXT, *p, lay.END],
                 [lay.DELETE, lay.END], [lay.DELETE, lay.TEXT, *p, lay.END]][int(rng.integers(6))])
        flush()
    return toks
