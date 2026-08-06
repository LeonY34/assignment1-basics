import regex as re

# text = "你好，world！我是大便。Hello, world! I am supertastylickinggoodyshit!"

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# s = re.findall(PAT, text)

# print(s)

# it = re.finditer(PAT, text)

# for x in it:
#     print(x.group(0), x.span())

# s = chr(0)
# print(s.__repr__())
# print("SDFSDF" + s + "SDFSDF")
# print(("SDFSDF" + s + "SDFSDF").__repr__())

def bpe(input_path: str, vocab_size: int, special_tokens: list[str])\
    -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Returns:
     - vocab: the tokenizer vocabulary (dict: int -> bytes);
     - merges: the merges performed along the process (list of pair of bytes).
    """
    if vocab_size < 256 + len(special_tokens):
        raise Exception("vocab_size too small!")
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, 'r') as f:
        text = f.read()
    if not text:
        raise Exception("Read Failed!")
    
    # protect special tokens
    for sp in special_tokens:
        text = text.replace(sp, f" {sp} ")
    
    vocab = {}
    indices = {}
    idx = 256
    for i in range(256):
        vocab[i] = bytes([i])
        indices[bytes([i])] = i
    for s in special_tokens:
        vocab[idx] = s.encode('utf-8')
        indices[s.encode('utf-8')] = idx
        idx += 1
    
    
    
    # print(vocab)
    
    it = re.finditer(PAT, text)
    
    source = list()
    
    pairs = {}
    
    for part in it:
        s = part.group(0)
        if s in special_tokens:
            source.append(indices[s.encode('utf-8')])
        else:
            for i, c in enumerate(s.encode('utf-8')):
                
                source.append(indices[c])
                if i != 0:
                    pairs[(source[-1], source[-2])] += 1
    
    print(pairs)
    # for x in it
    

if __name__ == "__main__":
    bpe("data/TinyStoriesV2-GPT4-valid.txt", 1000, ["<|endoftext|>"])
    
    
    
    