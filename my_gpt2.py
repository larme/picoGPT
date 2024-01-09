import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    max_x = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - max_x)
    base = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / base


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head, kv_cache=None): # [n_seq] -> [n_seq, n_vocab]
    # "wpe": [n_ctx, n_embd], positional embeddings
    # "wte": [n_vocab, n_embd], lookup table for vocab embeddings
    # "ln_f": {"b": [n_embd], "g": [n_embd]},
    # "blocks": [
    #     {
    #         "attn": {
    #             "c_attn": {"b": [3*n_embd], "w": [n_embd, 3*n_embd]},
    #             "c_proj": {"b": [n_embd], "w": [n_embd, n_embd]},
    #         },
    #         "ln_1": {"b": [n_embd], "g": [n_embd]},
    #         "ln_2": {"b": [n_embd], "g": [n_embd]},
    #         "mlp": {
    #             "c_fc": {"b": [4*n_embd], "w": [n_embd, 4*n_embd]},
    #             "c_proj": {"b": [n_embd], "w": [4*n_embd, n_embd]},
    #         },
    #     },

    # token + positional embeddings
    # wte[inputs] -> [n_seq, n_embd]]
    # wpe[range(len(inputs))] -> [[n_seq, n_embd]]

    if kv_cache is None:
        kv_cache = [None] * len(blocks)
        wpe_out = wpe[range(len(inputs))]
    else:
        wpe_out = wpe[[len(inputs) - 1]]
        inputs = [inputs[-1]]

    x = wte[inputs] + wpe_out  # [n_seq] -> [n_seq, n_embd]

    new_kv_cache = []
    for idx, block in enumerate(blocks):
        block_kv_cache = kv_cache[idx]
        x, updated_cache = transformer_block(x, **block, n_head=n_head, kv_cache=block_kv_cache)  # [n_seq, n_embd] -> [n_seq, n_embd]
        new_kv_cache.append(updated_cache)

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T, new_kv_cache  # [n_seq, n_embd] -> [n_seq, n_vocab]


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head, kv_cache=None):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # multi-head causal self attention
    attn_out, kv_cache_updated = mha(layer_norm(x, **ln_1), **attn, n_head=n_head, kv_cache=kv_cache)
    x = x + attn_out  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x, kv_cache_updated


def ffn(x, c_fc, c_proj):
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head, kv_cache=None): # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projections
    x = linear(x, **c_attn) # [n_seq, n_embd] @ [n_embd, 3*n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1) # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]

    # if kv_cache is passed, then append to them
    if kv_cache is not None:
        new_q, new_k, new_v = qkv # new_q, new_k, new_v = [1, n_embd]
        old_k, old_v = kv_cache
        k = np.vstack([old_k, new_k]) # k = [n_seq, n_embd], where n_seq = prev_n_seq + 1
        v = np.vstack([old_v, new_v]) # v = [n_seq, n_embd], where n_seq = prev_n_seq + 1
        qkv = [new_q, k, v]

    current_cache = [qkv[1], qkv[2]]

    # split into heads
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

    # causal mask to hide future inputs from being attended to
    if kv_cache:
        causal_mask = np.zeros((1, k.shape[0]))
    else:
        causal_mask = (1 - np.tri(x.shape[0])) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # merge heads
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj) # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]

    return x, current_cache


def generate(inputs, params, n_head, n_tokens_to_generate):
    kv_cache = None

    from tqdm import tqdm

    import time
    start = time.time()
    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits, kv_cache = gpt2(inputs, **params, n_head=n_head, kv_cache=kv_cache)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input
    end = time.time()
    delta = end - start
    print("tokens per seconds: ", n_tokens_to_generate/delta)

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids


def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    import fire

    fire.Fire(main)
