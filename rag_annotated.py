
"""
=============================================================================
 A MINIMAL RAG PIPELINE IN PLAIN PYTHON -- ANNOTATED VERSION
=============================================================================

 Every line of the working code is here, with the reasoning above it.

 SETUP
   pip install numpy sentence-transformers anthropic
   export ANTHROPIC_API_KEY="sk-ant-..."
   mkdir docs        # drop some .txt or .md files in here
   To run Script: python rag_annotated.py

 THE WHOLE PIPELINE IN ONE SENTENCE
   Cut documents into pieces -> turn each piece into a vector -> at question
   time turn the question into a vector too -> keep the pieces whose vectors
   point in the same direction -> paste those pieces into the prompt.

 Nothing else. Every RAG framework is elaboration on these six steps.
=============================================================================
"""

# --- os: needed for os.path.join and os.path.basename.
#     join builds paths portably ("docs/a.txt" on Linux, "docs\a.txt" on
#     Windows). basename strips the directory: "docs/paper.txt" -> "paper.txt",
#     which is what we want to show the user as a citation.
import os

# --- glob: finds files by wildcard pattern. glob("docs/*.txt") returns a list
#     of every .txt path in docs/. Without this you would have to hardcode
#     filenames, and adding a document would mean editing code.
import glob

# --- numpy: the entire "vector database" in this script is one numpy array.
#     We need it for two operations only: the matrix-vector product (which
#     computes all similarities at once) and argsort (which ranks them).
import numpy as np

# --- The embedding model. all-MiniLM-L6-v2 is 22M parameters, 384 output
#     dimensions, runs comfortably on CPU, and downloads in a few seconds.
#     It is the standard starting point. Bigger models (bge-large, e5-large,
#     gte-large) score better on benchmarks and cost more time per chunk;
#     swap the name string and nothing else changes.
from sentence_transformers import SentenceTransformer


# =============================================================================
# STEP 1 -- LOAD
# =============================================================================
# WHY THIS STEP EXISTS
#   Your knowledge lives in files, and every file format has a different way
#   of being read. A PDF needs pypdf, a Word file needs python-docx, an HTML
#   page needs BeautifulSoup. The loader's job is to hide all of that and hand
#   back ONE uniform shape so nothing downstream has to care.
#
#   That uniform shape is: {"text": <the words>, "source": <where it came from>}
#
# WHY "source" MATTERS AS MUCH AS "text"
#   This is metadata (metadata: data for data), and it is the difference between a system that says
#   "the learning rate was 1e-3" and one that says "the learning rate was
#   1e-3 [wavelets.txt]". Without metadata you cannot cite, cannot filter
#   ("only 2024 papers"), and cannot debug a bad answer back to its source.
#   Metadata is cheap to carry and impossible to add later. Carry it.

def load(folder="docs"):
    # Accumulator. We build a list rather than returning inside the loop
    # because we want every document, not the first one.
    docs = []

    # glob returns paths in arbitrary filesystem order, which means your chunk
    # IDs would shuffle between runs and make debugging miserable. sorted()
    # makes the pipeline deterministic: same folder in, same index out.
    # We concatenate two globs because glob has no "or" syntax -- one pattern
    # per extension. Add more patterns here as you add formats.
    # sorted() sort alphabetically, so "a.txt" comes before "b.txt" and "1.txt" comes before "2.txt".
    # "+" concatenates the two lists of paths into one list. 
    # Example: if folder contains "a.txt", "b.txt", "1.md", "2.md",
    # the result of the glob.glob calls will be ["a.txt", "b.txt"] + ["1.md", "2.md"] = ["a.txt", "b.txt", "1.md", "2.md"].
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))
                       + glob.glob(os.path.join(folder, "*.md"))):

        # encoding="utf-8" is not optional.  utf-8 is the standard encoding for text files, 
        # and it is the only encoding guaranteed to be portable across platforms. If you omit it,
        # then your code will work on your laptop but fail on the cluster with a UnicodeDecodeError,
        # because Python will guess the encoding from your locale. Always specify the encoding to avoid this issue.
        # Every real corpus (a corpus is just a collection of text documents) ends up
        # containing a few special characters somewhere -- a curly quote (“ or ” instead
        # of the plain ") or an accented name (José, François). You rarely notice them,
        # but they are the characters that break on a machine whose default encoding is
        # not utf-8. So always state the encoding.
        #
        # EXAMPLE -- a file holds bytes, never characters. utf-8 is the rule that turns
        # one into the other. Plain English letters cost 1 byte; anything else costs 2-4:
        #
        #     'Hello' -> 5 chars ->  5 bytes -> b'Hello'
        #     'José'  -> 4 chars ->  5 bytes -> b'Jos\xc3\xa9'   <- the é alone is 2 bytes
        #
        # The bytes on disk carry no label saying which encoding produced them, so the
        # reader has to be told. Same file, three different ways of reading it:
        #
        #     open(path, encoding="ascii")    -> UnicodeDecodeError on byte 0xc3
        #     open(path, encoding="latin-1")  -> "Paper by JosÃ© Hernández"  (no error!)
        #     open(path, encoding="utf-8")    -> "Paper by José Hernández"
        #
        # The middle case is the dangerous one: latin-1 maps every byte to something, so
        # nothing raises. The garbled text just flows on into your chunks and embeddings.

        # Question: utf-8 uses octal number system or hexadecimal number system, please clarify this first ?
        # Answer: Neither. UTF-8 works in binary — actual bits. Hexadecimal is only the notation people use 
        # to write those bits down, because one byte is exactly two hex digits, which is tidy.
        # Python's repr (representation) happens to print bytes as \xc3, and that \x prefix means "hex"
        # — that's the only reason hex shows up. Hence,
        # UTF-8 is defined in binary bit patterns. Hex is just how bytes get printed, 
        # and octal is an alternative notation nobody uses for this.

        # Remember: f.read() returns a string, not bytes. 
        # An encoding works in BOTH directions, and each direction has its own name:
        #     reading  bytes on disk (SSD/HDD) -> string in memory (RAM)  =  DECODING
        #     writing  string in memory (RAM)  -> bytes on disk (SSD/HDD) =  ENCODING
        # A CODEC (short for coder-decoder) is that two-way rule packaged under a name:
        # "utf-8", "latin-1", "ascii", "cp1252" are all codec names. One name covers both
        # directions, which is why the same "utf-8" works for reading and for writing.
        # The encoding= argument names the codec for whichever direction the file was
        # opened in. Here the mode is read, so it is only decoding -- but open(path, "w",
        # encoding="utf-8") would run the exact same table the other way. Outside of
        # files it is the str.encode() / bytes.decode() pair:
        #     'José'.encode('utf-8')          -> b'Jos\xc3\xa9'
        #     b'Jos\xc3\xa9'.decode('utf-8')  -> 'José'
        # Once it is a string, Python does not care about encoding anymore -- but only
        # for as long as it stays in memory. The moment you send it back out (write to a
        # file, a socket, a subprocess) an encoding gets chosen again, explicitly by you
        # or implicitly by the platform default. Same trap, other direction.
        with open(path, encoding="utf-8") as f:
            docs.append({"text": f.read(),
                         "source": os.path.basename(path)})
    return docs

# WHAT A REAL LOADER ADDS
#   For PDFs you would loop over pages, skip pages where extract_text()
#   returns None (those are scanned images and need OCR), strip the running
#   header that repeats on every page, and record the page number in the
#   metadata so citations can say "page 7". That is the "glue" a framework
#   sells you -- and it is also why loaders are the one part of LangChain
#   that people keep using even when they drop the rest.


# =============================================================================
# STEP 2 -- SPLIT (CHUNKING)
# =============================================================================
# WHY THIS STEP EXISTS -- two independent reasons
#
#   (a) SIZE. A 20-page paper does not fit in a prompt alongside everything
#       else you want to send.
#
#   (b) PRECISION -- the more important reason. An embedding is ONE vector for
#       the whole input, so it is a kind of average of everything in it. Embed
#       a 2000-word section covering charging, sleep mode, power profiles,
#       calibration, and disposal, and the "calibration" signal is roughly a
#       seventh of the vector's direction. A question about calibration then
#       matches it at ~0.6 instead of ~0.94, and loses to a shorter chunk that
#       is worse but more focused. Small chunks retrieve (retrieve means recover) better because their
#       vectors are undiluted.
#
# THE TRADE-OFF YOU ARE TUNING
#   size too small  -> the chunk is precise but incomplete. "Hold Fn+F5 for 10
#                      seconds" retrieves perfectly and omits the warning that
#                      makes it safe.
#   size too large  -> full context but diluted vector; retrieval starts missing.
#   So, start at 800 characters (~150-200 words) and adjust based on your data.
#
# WHY OVERLAP EXISTS
#   A hard cut lands wherever it lands, and sooner or later it lands in the
#   middle of the one sentence that answers the question:
#       chunk 1: "...decayed by a factor of 0.5 every"
#       chunk 2: "100 epochs. Training ran for 500 epochs..."
#   Neither chunk contains the complete fact, so neither embeds as an answer
#   to it. Overlap repeats the last ~150 characters of each chunk at the start
#   of the next, so any sentence shorter than the overlap survives intact
#   somewhere. Cost: ~20% more chunks to store and embed. Worth it.

def split(text, size=800, overlap=150):
    # Guard against a real bug I hit while testing this: if overlap >= size,
    # each iteration would advance `start` by only 1 character (because of the
    # max() on the last line), and a 2500-character string produced 2301
    # near-identical chunks. It does not hang, which makes it worse -- it
    # silently produces a garbage index. Fail loudly instead.
    # assert is python's built-in way to check a condition and raise an AssertionError if it is false. 
    # The syntax is assert <condition>, <error message>. If the condition is false,
    # the error message is printed and the program stops.
    assert overlap < size, "overlap must be smaller than size"

    # chunks: the output. start: a moving cursor into `text`.
    chunks, start = [], 0

    # Walk the cursor forward until it runs off the end of the string.
    while start < len(text):

        # Tentative end of this chunk. min() clamps it so the final chunk does
        # not index past the end of the string.
        end = min(start + size, len(text))

        # Only look for a nicer break point if we are NOT at the last chunk.
        # (At the end of the text there is nothing to break cleanly against.)
        if end < len(text):

            # rfind searches BACKWARDS for the last space in the window
            # [start + size//2, end). Meaning: "give up as much as half a chunk
            # to avoid cutting a word in half, but no more than that."
            #
            #   without this:  "...the Fourier Neural Oper" | "ator implements..."
            #   with this:     "...the Fourier Neural"      | "Operator implements..."
            #
            # The lower bound matters. Without it, a text with one space near
            # the beginning would collapse chunks down to almost nothing.
            #
            # ----------------------------------------------------------------
            # PROOF -- a 20-character text, size=8, overlap=2:
            #
            #     text  = 'Fi2 ATGCATGCATGCATGC'
            #     index    01234567890123456789
            #                 ^
            #                 the ONLY space, at index 3
            #
            # Realistic version of this: a short label followed by something
            # with no spaces in it -- a hash, a URL, a DNA read, a base64 blob.
            #
            # WITH the lower bound -- rfind(" ", start + size//2, end)
            # size//2 is 4, so the search window starts at start + 4.
            #
            #   start=0   end=8    rfind(" ", 4, 8)  = -1     chunk = 'Fi2 ATGC'   len 8
            #   start=6   end=14   rfind(" ", 10, 14) = -1    chunk = 'GCATGCAT'   len 8
            #   start=12  end=20   (last chunk, no search)    chunk = 'ATGCATGC'   len 8
            #
            # 3 chunks, all full size. On the first pass the window is [4, 8) --
            # the space at index 3 is outside it, so rfind returns -1 and the
            # code keeps the hard cut at 8.
            #
            # WITHOUT it -- rfind(" ", start, end)
            #
            #   start=0   end=8    rfind(" ", 0, 8)  = 3  -> end=3   chunk = 'Fi2'   len 3
            #   start=1   end=9    rfind(" ", 1, 9)  = 3  -> end=3   chunk = 'i2'    len 2
            #   start=2   end=10   rfind(" ", 2, 10) = 3  -> end=3   chunk = '2'     len 1
            #   start=3   end=11   rfind(" ", 3, 11) = 3  -> end=3   chunk = ''      DROPPED
            #   start=4   end=12   rfind(" ", 4, 12) = -1           chunk = 'ATGCATGC'  len 8
            #   start=10  end=18   rfind(" ", 10, 18) = -1          chunk = 'GCATGCAT'  len 8
            #   start=16  end=20   (last chunk)                     chunk = 'ATGC'      len 4
            #
            # 6 chunks: ['Fi2', 'i2', '2', 'ATGCATGC', 'GCATGCAT', 'ATGC']
            #
            # THE MECHANISM, LINE BY LINE
            # Look at the rfind column in the second trace: 3, 3, 3, 3. It keeps
            # finding the same space.
            #
            # Pass 1 finds index 3 and sets end = 3, so the chunk is only 3
            # characters instead of 8. Now the cursor advance runs:
            #
            #     start = max(end - overlap, start + 1)
            #           = max(3 - 2, 0 + 1)
            #           = max(1, 1) = 1
            #
            # end - overlap is 1, so the cursor moves forward by one character,
            # not by the ~6 you would expect. That max(..., start + 1) is the
            # only thing keeping this from being an infinite loop -- it
            # guarantees progress, but only the bare minimum.
            #
            # On pass 2, start is 1 and the window is [1, 9). Index 3 is still
            # inside it. Same space, found again, end = 3 again, chunk 'i2'. And
            # so on: each pass emits a one-character-shorter suffix of the same
            # four characters, crawling forward until start finally passes index
            # 3 and the space drops out of the window. Only then does chunking
            # recover to full-size pieces.
            #
            # The lower bound breaks this at the source. Starting the search
            # half a chunk in means the window can never contain a position the
            # cursor has almost reached -- so a space cannot be found twice, and
            # end can never land closer than size/2 from start. Worst case rfind
            # returns -1 and you get a hard cut mid-word, which is the
            # acceptable outcome described above: give up at most half a chunk
            # to avoid splitting a word, no more.
            #
            # The junk is not free, either. 'Fi2', 'i2' and '2' each get
            # embedded into a 384-dimensional vector, each take a row in the
            # similarity matrix, and each can be retrieved and pasted into a
            # prompt -- carrying no information at all.
            # ----------------------------------------------------------------
            cut = text.rfind(" ", start + size // 2, end)

            # rfind returns -1 when there is no space in the window at all --
            # e.g. a long base64 blob or a minified line. In that case we keep
            # the hard cut rather than crashing or looping forever.
            if cut != -1:
                end = cut

        # Slice out the chunk. .strip() removes the leading space left behind
        # by cutting at a space, plus any stray newlines at the edges.

        piece = text[start:end].strip()

        # A chunk can be empty (e.g. the text had a run of blank lines).
        # Empty strings still get embedded into a vector, still occupy a row
        # in the matrix, and can still be retrieved -- returning nothing useful
        # to the LLM. Drop them here.
        if piece:
            chunks.append(piece)

        # We have consumed the whole string; stop. Without this the next line
        # would move `start` backwards by `overlap` and we would re-chunk the
        # tail of the document forever.
        if end >= len(text):
            break

        # Advance the cursor, stepping BACK by `overlap` -- this is what makes
        # consecutive chunks share text. max(..., start + 1) is a safety net
        # guaranteeing forward progress even in degenerate cases.
        start = max(end - overlap, start + 1)

    return chunks

# BETTER SPLITTERS, WHEN YOU NEED THEM
#   Recursive character splitting: try to break on "\n\n" first, then "\n",
#     then ". ", then " ". Respects paragraph structure instead of just words.
#     This is what LangChain's default splitter does; it is ~40 more lines.
#   Structure-aware: split on Markdown headers or LaTeX \section boundaries,
#     so a chunk is always one coherent unit.
#   Semantic: embed each sentence, cut where consecutive sentences diverge.
#     Elegant, and expensive -- an embedding call per sentence.


# =============================================================================
# STEP 3 -- EMBED (VECTORIZATION)
# =============================================================================
# WHY THIS STEP EXISTS
#   Computers cannot compare meanings, only numbers. An embedding model maps
#   text to a point in 384-dimensional space such that texts with similar
#   meaning land near each other. That property is what makes the search
#   semantic rather than lexical:
#
#     "how do I reset my password"  and  "I forgot my login"
#
#   share almost no words -- keyword search scores them near zero -- but their
#   embeddings are close, so cosine similarity scores them high.
#
# THE ONE RULE YOU MUST NOT BREAK
#   Documents and queries must be embedded by the SAME model. Different models
#   produce different, unrelated coordinate systems. A vector from MiniLM
#   (384-dim) and one from OpenAI (1536-dim) are not comparable -- and even if
#   the dimensions matched, the axes would mean different things. Changing the
#   embedding model means re-embedding your entire corpus from scratch. This
#   is the one migration no framework can save you from.

# Loaded ONCE at import, not inside embed(). Constructing a SentenceTransformer
# reads several hundred MB of weights off disk into RAM. Doing that per call
# would make every query take seconds instead of milliseconds.
encoder = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    # NOTE: takes a LIST, returns a matrix of shape (len(texts), 384).
    # Batching matters -- one forward pass over 64 texts is far faster than 64
    # separate passes, because the GPU/CPU is doing one big matrix multiply
    # instead of 64 small ones.
    #
    # normalize_embeddings=True rescales every output vector to unit length
    # (||v|| = 1). This is a genuine optimisation, not cosmetic. Cosine
    # similarity is
    #
    #       cos(q, d) = (q . d) / (||q|| ||d||)
    #
    # and when both vectors already have norm 1 the denominator is 1, so
    #
    #       cos(q, d) = q . d
    #
    # A plain dot product. That is why step 5 can be a single `matrix @ q`
    # with no division anywhere -- the normalisation was done here, once,
    # instead of on every query forever.
    return encoder.encode(texts, normalize_embeddings=True, batch_size=64)


# =============================================================================
# STEP 4 -- INDEX (BUILD THE "VECTOR DATABASE")
# =============================================================================
# WHY THIS STEP EXISTS
#   Steps 1-3 are per-document work. This step runs them over everything and
#   parks the result in memory so queries are cheap. In production this is an
#   offline batch job; here it runs at startup.
#
# WHAT A VECTOR DATABASE ACTUALLY IS
#   Two things kept in sync:
#     - `matrix`: the searchable numbers, shape (N_chunks, 384)
#     - `chunks`: the human-readable text + metadata, same order
#   Row i of the matrix corresponds to chunks[i]. That positional correspondence
#   IS the index. FAISS, Qdrant and Chroma add persistence, metadata filtering,
#   and an ANN graph so search is O(log N) instead of O(N) -- but conceptually
#   they are still just these two aligned collections.
#
#   Which means: you do not need a vector database to start. Up to roughly
#   50,000 chunks, numpy brute force answers in a few milliseconds. Reach for
#   FAISS when you measure a problem, not before.

# One flat list comprehension doing two nested loops:
#   for each document -> for each chunk of that document -> make a record.
# Note that `d["source"]` is copied onto every chunk. This is the link that
# lets the answer cite a filename. Losing it here means you can never recover
# it later -- the matrix rows are anonymous numbers.
chunks = [{"text": c, "source": d["source"]}
          for d in load() for c in split(d["text"])]

# The single most expensive line in the file. Every chunk goes through the
# transformer once. For a few hundred chunks this is seconds; for a million it
# is hours, which is why real systems cache this to disk (np.save) and only
# re-embed documents that changed.
#
# We pass a list comprehension pulling out just the text -- the model wants
# strings, not dicts.
matrix = embed([c["text"] for c in chunks])

# Not decoration. This one line catches the three most common silent failures:
#   "indexed 0 chunks"        -> your docs/ folder is empty or misnamed
#   "indexed 1 chunks"        -> your splitter is broken
#   shape[1] not what expected-> you are not running the model you think
print(f"indexed {len(chunks)} chunks, matrix shape {matrix.shape}")


# =============================================================================
# STEP 5 -- RETRIEVE
# =============================================================================
# WHY THIS STEP EXISTS
#   This is the "R" in RAG. Given a question, find the handful of chunks most
#   likely to contain the answer. Everything before this was preparation.

def retrieve(query, k=4):
    # Embed the question with the SAME encoder used for the documents.
    # embed() expects a list and returns a matrix, so we pass [query] and take
    # row [0] to get back a single 384-vector.
    q = embed([query])[0]

    # THE HEART OF THE WHOLE PIPELINE, and it is one matrix-vector product.
    #
    #   matrix is (N, 384),  q is (384,),  so matrix @ q is (N,)
    #
    # Row i of the result is the dot product of chunk i's vector with the
    # query's vector -- which, because both are unit length (step 3), IS the
    # cosine similarity. One BLAS call computes all N similarities at once;
    # a Python loop over N chunks would be ~100x slower.
    #
    # Worked example with 2D vectors, to make it concrete:
    #   q  = (0.95, 0.30)   "how do I feed a kitten"
    #   d1 = (1.00, 0.00)   "cats"     -> 0.95*1.00 + 0.30*0.00 = 0.950
    #   d2 = (0.90, 0.40)   "kittens"  -> 0.95*0.90 + 0.30*0.40 = 0.975  <- wins
    #   d3 = (0.00, 1.00)   "dogs"     -> 0.95*0.00 + 0.30*1.00 = 0.300
    scores = matrix @ q

    # argsort returns the INDICES that would sort the array, ascending.
    # Negating the scores flips it to descending, so we get the best first.
    # [:k] keeps only the top k.
    #
    #   scores = [0.950, 0.975, 0.300]
    #   -scores           -> [-0.950, -0.975, -0.300]
    #   np.argsort(...)   -> [1, 0, 2]        (d2 best, then d1, then d3)
    #   [:2]              -> [1, 0]
    #
    # For large N, np.argpartition(-scores, k)[:k] is faster -- it does a
    # partial sort in O(N) instead of a full sort in O(N log N). Not worth the
    # extra complexity until N is large.
    top = np.argsort(-scores)[:k]

    # Return the text records, not the row numbers -- the caller needs the
    # text and the source. We return the score alongside because it is the
    # single most useful debugging signal you have: if your top hit scores
    # 0.28, retrieval failed and the LLM is about to hallucinate. Watch these
    # numbers. On MiniLM, a good match is usually >0.5.
    #
    # float() converts numpy's float32 to a normal Python float, which prints
    # cleanly and serialises to JSON without complaint.
    return [(chunks[i], float(scores[i])) for i in top]

# HOW k IS A REAL TRADE-OFF
#   too small -> the answer simply is not in the context; the model cannot
#                recover from that, no matter how good it is
#   too large -> more noise, more tokens, more cost, and the "lost in the
#                middle" effect where models under-attend to the centre of a
#                long prompt
#   k=4 is a sane default for 800-character chunks.
#
# WHAT TO ADD FIRST, WHEN THIS IS NOT GOOD ENOUGH
#   1. Hybrid search: run BM25 keyword search alongside this and merge the two
#      ranked lists. Dense search misses exact strings -- ask about the error
#      code CUDA_ERR_7734 and cosine similarity has no idea, because the code
#      carries no "meaning". Biggest easy win available.
#   2. Reranking: retrieve k=50 here, then score each one properly with a
#      cross-encoder and keep the best 4. Second biggest win.
#   Both are additions to this function. Neither requires a framework.


# =============================================================================
# STEP 6 -- GENERATE
# =============================================================================
# WHY THIS STEP EXISTS
#   The "AG" in RAG. We hand the retrieved text to the model and ask it to
#   answer FROM that text rather than from its training weights.

# The prompt template carries three instructions, each earning its place:
#
#   "ONLY the context below"       -- without it the model blends retrieved
#                                     text with half-remembered training data,
#                                     and you cannot tell which is which.
#   "say you don't know"           -- gives the model a licence to decline.
#                                     Without an explicit escape hatch, a model
#                                     handed irrelevant context will still
#                                     produce a confident-sounding answer,
#                                     because that is what the prompt implies
#                                     it should do.
#   "Cite the chunk number"        -- makes the answer auditable. You can check
#                                     whether [2] actually says what the model
#                                     claims. This is the practical advantage
#                                     RAG has over fine-tuning, and you only
#                                     get it if you ask for it.
TEMPLATE = """Answer the question using ONLY the context below.
If the context does not contain the answer, say you don't know.
Cite the chunk number you used, like [2].

Context:
{context}

Question: {question}
Answer:"""

def answer(query, k=4):
    # Step 5 gives us the raw material.
    hits = retrieve(query, k)

    # Flatten the retrieved records into one string for the prompt.
    #
    # enumerate(...) supplies i so we can number the chunks [1], [2], ... --
    # the model needs a label to cite. (i+1 because humans count from 1.)
    #
    # The source filename goes in too, so the model can attribute to a document
    # and so YOU can see, when reading the prompt in a log, where each block
    # came from.
    #
    # The blank line between chunks ("\n\n") is not cosmetic: it signals to the
    # model that these are separate, unrelated passages rather than one
    # continuous text. Without a clear separator, models sometimes read across
    # a boundary and merge two unrelated facts into one wrong sentence.
    #
    # `_` discards the score -- useful for us, noise for the model.
    context = "\n\n".join(
        f"[{i+1}] (source: {h['source']})\n{h['text']}"
        for i, (h, _) in enumerate(hits))

    # Imported here rather than at the top so the retrieval half of this script
    # runs with no API key at all. Try `retrieve("your question")` on its own
    # first -- if retrieval is bad, no LLM can rescue the answer, and you will
    # waste a lot of time blaming the wrong component.
    import anthropic

    # Reads ANTHROPIC_API_KEY from the environment automatically. Never paste a
    # key into source -- it ends up in git history, and secret scanners on
    # GitHub will find it within minutes.
    client = anthropic.Anthropic()

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        # Required by this API (optional in OpenAI's) -- an example of exactly
        # the cross-provider difference that makes swapping models annoying,
        # and that a framework wrapper papers over.
        max_tokens=1000,
        # .format() substitutes {context} and {question} in TEMPLATE.
        # If your documents contain literal { or } characters, .format() will
        # raise KeyError -- switch to an f-string or str.replace() then.
        messages=[{"role": "user",
                   "content": TEMPLATE.format(context=context,
                                              question=query)}])

    # The response is a LIST of content blocks, not a string -- the API can
    # return text blocks, tool-use blocks, and images in one message. For a
    # plain text reply there is one block, so [0].text is the answer.
    # (For comparison: OpenAI needs resp.choices[0].message.content, and Gemini
    # needs resp.candidates[0].content.parts[0].text. Same idea, three shapes.)
    return msg.content[0].text


# =============================================================================
# STEP 7 -- RUN
# =============================================================================
# This guard means the file can be imported ("from rag import retrieve") for
# experiments in a notebook WITHOUT starting the interactive loop. Without it,
# importing the module would hang waiting for keyboard input.
if __name__ == "__main__":
    while True:
        q = input("\nquestion (or 'quit'): ").strip()

        # Empty input catches a bare Enter press, which would otherwise embed
        # an empty string and retrieve nonsense.
        if q in {"quit", "exit", ""}:
            break

        # Print the retrieved chunks and their scores BEFORE the answer.
        # This is the most valuable habit in the whole script: when an answer is
        # wrong you immediately see which of the two halves failed.
        #   bad chunks + bad answer  -> fix retrieval (chunk size, k, hybrid)
        #   good chunks + bad answer -> fix the prompt or the model
        # Without this, every failure looks identical and you debug blind.
        for h, s in retrieve(q):
            print(f"  {s:.3f}  {h['source']}  {h['text'][:70]}...")

        print("\n" + answer(q))


# =============================================================================
# WHAT LANGCHAIN WOULD HAVE ADDED -- AND WHAT IT WOULD HAVE HIDDEN
# =============================================================================
# The equivalent LangChain script is about 12 lines. What those 12 lines buy:
#
#   + 100+ document loaders, already handling the PDF edge cases
#   + one interface over ~80 vector databases
#   + retry/backoff, batching, and rate-limit handling on API calls
#   + provider swapping without touching your code
#   + every retrieval strategy (hybrid, MMR, reranking, parent-document)
#     pre-implemented behind the same .invoke() call
#
# What they hide, which you have now seen:
#
#   - that chunking is a lossy decision with a real accuracy trade-off
#   - that normalisation is why similarity is a dot product
#   - that the "vector database" is two aligned lists
#   - that retrieval scores exist and are your primary debugging signal
#   - that the prompt template is doing real work, and is tunable
#
# The judgement you can now make: use the loaders (that ecosystem is genuinely
# hard to replicate), and consider writing the core loop yourself. Many
# production teams land exactly there.
# =============================================================================
