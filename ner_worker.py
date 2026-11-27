"""One process = one (GPU, model) pair.

Running all 3 models in parallel on each GPU (6 processes total) keeps the GPUs
busy while other processes do CPU work (tokenization, Doc building, entity
extraction), instead of each GPU running salary -> positions -> requirements
one after another.
"""
import os

MODELS = {
    "salary":       "salary_model",
    "positions":    "positions_model",
    "requirements": "requirements_model",
}

# Components NER never reads from. If the models were trained on top of a
# full pipeline (e.g. en_core_web_trf), these still run on every doc unless
# turned off. Things like entity_ruler are deliberately NOT here because they
# change NER output.
_UNUSED = {
    "tagger", "morphologizer", "parser", "lemmatizer", "trainable_lemmatizer",
    "attribute_ruler", "senter", "textcat", "textcat_multilabel",
}


def _batches(texts, max_chars, max_docs):
    """Batch by total characters, not doc count, so GPU memory per batch stays
    roughly constant: many short texts per batch, few long ones."""
    batch, size = [], 0
    for t in texts:
        if batch and (size + len(t) > max_chars or len(batch) >= max_docs):
            yield batch
            batch, size = [], 0
        batch.append(t)
        size += len(t)
    if batch:
        yield batch


def run_model(gpu_id, name, texts, progress_queue=None,
              max_chars=500_000, max_docs=4096, cpu_threads=4):
    # 6 processes on one node: stop each from grabbing every CPU core.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(cpu_threads)
    try:
        import torch
        torch.set_num_threads(cpu_threads)
        # TF32 matmuls: free speedup on Blackwell/Ampere+, negligible accuracy impact.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except ImportError:
        pass

    import spacy
    spacy.require_gpu(gpu_id)
    nlp = spacy.load(MODELS[name])
    for pipe in [p for p in nlp.pipe_names if p in _UNUSED]:
        nlp.disable_pipe(pipe)
    labels = list(nlp.get_pipe("ner").labels)

    out = []
    for batch in _batches(texts, max_chars, max_docs):
        for doc in nlp.pipe(batch, batch_size=len(batch)):
            ents = {}
            for ent in doc.ents:
                ents.setdefault(ent.label_, []).append(
                    (ent.start_char, ent.end_char, ent.text)
                )
            out.append(ents)
        if progress_queue is not None:
            progress_queue.put((gpu_id, name, len(batch)))

    if progress_queue is not None:
        progress_queue.put((gpu_id, name, None))   # done signal
    # nlp.pipe_names is a spaCy SimpleFrozenList, which can't be unpickled in
    # the parent process -> convert to a plain list.
    return out, labels, list(nlp.pipe_names)

