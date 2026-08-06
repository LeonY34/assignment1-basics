import regex

class bpeTokenizer:
    ranks: dict[bytes, int] = {}
    # tokens: list[bytes] = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    def __init__(self, data: str, vocab_size: int, special_tokens: list[str]=['<|endoftext|>']):
        self.train(data, vocab_size, special_tokens)
    
    def train(self, data: str, vocab_size: int, special_tokens: list[str]) -> dict[bytes, int]:
        """
        Byte-level bpe train
        """
        
        if vocab_size < 256:
            raise Exception("vocab_size too small.")
        
        # protect special tokens
        for sp in special_tokens:
            data = data.replace(sp, f" {sp} ")
            
        # init
        ranks: dict[bytes, int] = {bytes([i]) : i for i in range(256)}
        # tokens = [bytes([i]) for i in range(256)]
            
        # divide
        words = [word.encode('utf-8') for word in regex.findall(self.PAT, data)]
        
        print(words)
        
        # convert into bytes
        word_parts: list[list[bytes]] = [
            [bytes([b]) for b in word] for word in words   
        ]
        
        while len(ranks) < vocab_size:
            cnt: dict[bytes, int] = {}
            
            for word_part in word_parts:
                for pair in zip(word_part[:-1], word_part[1:]):
                    pair_bytes = pair[0] + pair[1]
                    # if not found, return 0
                    cnt[pair_bytes] = cnt.get(pair_bytes, 0) + 1

            if not cnt:
                break
            
            maxpair = max(cnt, key=cnt.get)
            if cnt[maxpair] == 0:
                break
            ranks[maxpair] = len(ranks)
            # tokens += [maxpair]
            
            for idx, word_part in enumerate(word_parts):
                if len(word_part) < 2:
                    continue

                merged = [word_part[0]]
                for token in word_part[1:]:
                    if merged[-1] + token == maxpair:
                        merged[-1] = merged[-1] + token
                    else:
                        merged.append(token)
                word_parts[idx] = merged
                
        self.ranks = ranks
        # self.tokens = tokens
            
    def __encode(self, word: list[bytes]) -> list[int]:
        """
        O(nm) implementation.
        for smaller words
        """
        ranks = self.ranks
        
        while True:
            mink = len(ranks)
            for i, pair in enumerate(zip(word[:-1], word[1:])):
                pr = pair[0] + pair[1]
                rk = ranks.get(pr, -1)
                if rk != -1 and rk < mink:
                    mink = rk
                    pos = i
            if mink == len(ranks):
                break

            word = word[:pos] + [word[pos] + word[pos + 1]] + word[pos+2: ]
        
        # print(f"word: {word}")
        return [ranks[i] for i in word]
                    
    def encode(self, text: str) -> list[int]:
        res = []
        words = [[bytes([i]) for i in s.encode('utf-8')] for s in regex.findall(self.PAT, text)]  
        
        for word in words:
            # print(f"processing {word}")
            res += self.__encode(word)
        
        return res
        
    
    def decode():
        pass
    
if __name__ == "__main__":
    text = r"This story, \"Report: Google on pace to sell 3 million Pixels by the end of the year\" was originally published by Greenbot .<|endoftext|>"
    tokenizer = bpeTokenizer(text, 300, [])
    print(tokenizer.ranks)
    
    x = "I am dog."
    res = tokenizer.encode(x)
    print(res)
    print(type(res))
    print(type(res[0]))
    # with open('/Users/leon34/Desktop/CSdiy/stanfordCS336/Code/Lab1/assignment1-basics/data/owt_train.txt', 'r') as f:
    #     for i in range(100):
    #         s = f.readline()
    #         print(s)