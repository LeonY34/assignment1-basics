from __future__ import annotations
from copy import copy
import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool
from collections.abc import Iterable
from typing import TypeVar
import time
import heapq
import pickle
# import json

T = TypeVar("T")

class LinkedListNode:
    """
        LinkedListNode
    """
    pre: LinkedListNode | None
    nxt: LinkedListNode | None
    value: T | None
    merge_op: function | None

    def __init__(self, value: T | None = None, pre: LinkedListNode | None = None, nxt: LinkedListNode | None = None, merge_op: function | None = None):
        self.value = value
        self.pre = pre
        self.nxt = nxt
        self.merge_op = merge_op
        
    def merge_with_nxt(self):
        assert self.nxt != None
        if self.merge_op == None:
            self.value += self.nxt.value
        else:
            self.value = self.merge_op(self.value, self.nxt.value)
        self.nxt.value = None
        self.nxt = self.nxt.nxt
        if self.nxt != None:
            self.nxt.pre = self

class LinkedListIterator:
    """
        Iterator for LinkedList
    """
    
    current: LinkedListNode[T] | None
    
    def __init__(self, current):
        self.current = current
    
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.current == None:
            raise StopIteration
        
        now = self.current
        self.current = self.current.nxt
        return now

class LinkedList:
    """
        LinkedList
    """
    
    head: LinkedListNode[T] | None
    tail: LinkedListNode[T] | None
    def __init__(self, l: Iterable[T] | None = None, merge_op: function | None = None):
        self.head = None
        self.tail = None
        if l != None:
            for v in l:
                # print(v)
                now = LinkedListNode(v, self.tail, None, merge_op)
                if self.head == None: 
                    self.head = now
                else:
                    self.tail.nxt = now
                self.tail = now
                
                    
    def __iter__(self):
        return LinkedListIterator(self.head)
        

class WordNode:
    """
        WordNode
        All merged/single bytes are represented by integers
    """
    
    word: bytes
    word_position: int
    times: int
    is_big_node: bool
    linked_tokens: LinkedList[LinkedListNode[bytes]] # used when word is long
    arrayed_tokens: list[bytes] # used when word is small
    
    init_counts: dict[tuple[bytes, bytes], list | int] # only true in init step
    
    BIG_WORD_THRESHOLD = 100
    
    def __init__(self, word: bytes, word_position: int, times: int):
        self.word = word
        self.word_position = word_position
        self.times = times
        self.is_big_node = len(word) >= self.BIG_WORD_THRESHOLD
        # self.is_big_node = True # 现在small node有问题，暂时抛弃
        self.arrayed_tokens = [bytes([i]) for i in word]
        # self.init_counts = defaultdict(lambda: [0, []])
        self.init_counts = {}
        if self.is_big_node:
            self.linked_tokens = LinkedList(self.arrayed_tokens)
            now = self.linked_tokens.head
        for i in range(len(self.arrayed_tokens) - 1):
            tp = (self.arrayed_tokens[i], self.arrayed_tokens[i + 1])
            if tp not in self.init_counts:
                if self.is_big_node:
                    self.init_counts[tp] = [1, [now]]
                else:
                    self.init_counts[tp] = 1
            else:
                if self.is_big_node:
                    self.init_counts[tp][0] += 1
                    self.init_counts[tp][1].append(now)
                else:
                    self.init_counts[tp] += 1
            if self.is_big_node:
                now = now.nxt
    
    def __str__(self):
        return f"WordNode(word: {self.word}, times: {self.times}, is_big_node: {self.is_big_node}, arrayed_tokens: {self.arrayed_tokens})"
    
    def __repr__(self):
            return f"WordNode(word: {self.word}, times: {self.times}, is_big_node: {self.is_big_node}, arrayed_tokens: {self.arrayed_tokens})"
    
    def update( # 这个可能会成为瓶颈
        self, 
        list_nodes: list[LinkedListNode[bytes] | int], 
        byte_pair: tuple[bytes, bytes], 
        cache_times: dict[tuple[bytes, bytes], PairNode]
        ) -> set[tuple[bytes, bytes]]:
        
        inserts: dict[tuple[bytes, bytes], list | int] = {}
        decreases: set[tuple[bytes, bytes]] = set()
        merged_bytes = byte_pair[0] + byte_pair[1]
        
        if self.is_big_node:
            for nd in list_nodes:
                if nd.value == byte_pair[0] and nd.nxt != None and nd.nxt.value == byte_pair[1]:
                    if nd.pre != None:
                        if nd.pre.value != merged_bytes:
                            tp = (nd.pre.value, nd.value)
                            cache_times[tp].times -= self.times
                            decreases.add(tp)
                        tp2 = (nd.pre.value, merged_bytes)
                        if tp2 not in inserts: inserts[tp2] = [nd.pre]
                        else: inserts[tp2].append(nd.pre)
                    if nd.nxt.nxt != None:
                        tp = (nd.nxt.value, nd.nxt.nxt.value)
                        cache_times[tp].times -= self.times
                        decreases.add(tp)
                        if nd.nxt.nxt.nxt == None or (nd.nxt.nxt.value, nd.nxt.nxt.nxt.value) != byte_pair:
                            tp2 = (merged_bytes, nd.nxt.nxt.value)
                            if tp2 not in inserts: inserts[tp2] = [nd]
                            else: inserts[tp2].append(nd)
                    nd.merge_with_nxt()
        else:
            arrayed_tokens_new = []
            i = 0
            while i < len(self.arrayed_tokens):
                if i + 1 == len(self.arrayed_tokens):
                    arrayed_tokens_new.append(self.arrayed_tokens[i])
                    break
                p1, p2 = self.arrayed_tokens[i], self.arrayed_tokens[i + 1]
                tp = (p1, p2)
                if tp == byte_pair:
                    arrayed_tokens_new.append(p1 + p2)
                    if i >= 1:
                        tp1 = (self.arrayed_tokens[i - 1], self.arrayed_tokens[i])
                        cache_times[tp1].times -= self.times
                        decreases.add(tp1)
                        tp2 = (arrayed_tokens_new[-2], merged_bytes)
                        if tp2 not in inserts: inserts[tp2] = 1
                        else: inserts[tp2] += 1
                    if i + 2 < len(self.arrayed_tokens):
                        tp1 = (self.arrayed_tokens[i + 1], self.arrayed_tokens[i + 2])
                        cache_times[tp1].times -= self.times
                        decreases.add(tp1)
                        if i + 3 >= len(self.arrayed_tokens) or (self.arrayed_tokens[i + 2], self.arrayed_tokens[i + 3]) != byte_pair:
                            tp2 = (merged_bytes, self.arrayed_tokens[i + 2])
                            if tp2 not in inserts: inserts[tp2] = 1
                            else: inserts[tp2] += 1
                    i += 2
                else:
                    arrayed_tokens_new.append(self.arrayed_tokens[i])
                    i += 1
            self.arrayed_tokens = arrayed_tokens_new
                    
        for key, value in inserts.items():
            if key not in cache_times:
                if self.is_big_node:
                    cache_times[key] = PairNode([self.word_position], [value], self.times * len(value), key)
                else:
                    cache_times[key] = PairNode([self.word_position], [None], self.times * value, key)
            else:
                cache_times[key].word_positions.append(self.word_position)
                cache_times[key].in_word_nodes.append(value if self.is_big_node else None)
                cache_times[key].times += self.times * len(value) if self.is_big_node else self.times * value
        
        return set(inserts.keys()) | decreases
                    
class PairNode:
    """
        Pair Node for Heap queue
    """
    
    word_positions: list[int]
    in_word_nodes: list[list[LinkedListNode[bytes]]] | None
    times: int
    byte_pair: tuple[bytes, bytes]
    version: int
    
    def __init__(
        self, 
        word_positions: list[int], 
        in_word_nodes: list[list[LinkedListNode[bytes] | int]] | None, 
        times: int, 
        byte_pair: tuple[bytes, bytes],
        version: int = 0):
        
        self.word_positions = word_positions
        self.in_word_nodes = in_word_nodes
        self.times = times
        self.byte_pair = byte_pair
        self.version = version
    
    def __lt__(self, other: PairNode):
        return (self.times, self.byte_pair) > (other.times, other.byte_pair)

    def __str__(self):
        return f"PairNode(word_posisionts: {self.word_positions}, in_word_nodes: {self.in_word_nodes}, times: {self.times}, bytes_pair: {self.byte_pair})"

    def __repr__(self):
            return f"PairNode(word_posisionts: {self.word_positions}, in_word_nodes: {self.in_word_nodes}, times: {self.times}, bytes_pair: {self.byte_pair})"
    

class BPEtokenizer:
    """
        BPE Tokenizer
    """
    vocab_size: int
    cur_size: int
    vocab_map: dict[int, bytes]
    merge_ranks: dict[tuple[bytes, bytes], int]
    special_words: list[str]
    training_backend: str
    inference_backend: str
    train_filename: str
    store_filename: str
    parallel_num: int
    
    cache_heap: list[PairNode]
    cache_words: list[WordNode]
    cache_times: dict[tuple[bytes, bytes], PairNode]
    
    # TRAINING_BACKENDS = ["bruteforce", "heap", "hybrid"]
    # INFERENCE_BACKENDS = ["bruteforce", "heap", "hybrid"]
    TRAINING_BACKENDS = ["heap"]
    INFERENCE_BACKENDS = ["bruteforce"]
    MINI_CHUNK_SIZE = 4096
    PRE_TOKENIZE_CHUNK_SIZE = 1024 * 1024
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    @classmethod
    def find_chunk_boundaries(
        cls,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(cls.MINI_CHUNK_SIZE)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += cls.MINI_CHUNK_SIZE

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))
    
    def __init__(
        self, 
        load_filename: str = None,
        
        vocab_size: int = None, 
        special_words: list[str] = None, 
        training_backend: str = "heap", 
        inference_backend: str = "bruteforce",
        parallel_num: int = 4):
        
        if load_filename != None:
            self.load_from_file(load_filename)
            return
        
        self.vocab_size = vocab_size
        self.cur_size = 0
        self.special_words = special_words
        self.parallel_num = parallel_num
        if training_backend not in self.TRAINING_BACKENDS: raise Exception(f"Available training backends: {self.TRAINING_BACKENDS}")
        if inference_backend not in self.INFERENCE_BACKENDS: raise Exception(f"Available inference backends: {self.INFERENCE_BACKENDS}")
        self.training_backend = training_backend
        self.inference_backend = inference_backend
    
        
    def pre_tokenize_no_special(self, text: str) -> list[bytes]:
        matches = re.finditer(self.PAT, text)
        return [match.group().encode('utf-8') for match in matches]
    
    def _pre_tokenize_from_file_place(self, filename: str, rg: tuple[int, int]) -> dict[bytes, int]:
        print(f"start from {rg[0]} to {rg[1]}...", flush=True)
        st_time = time.time()
        with open(filename, "rb") as f:
            f.seek(rg[0])
            # text = f.read(rg[1] - rg[0]).decode('utf-8')
            words: dict[bytes, int] = {}
            p = rg[0]
            pre_part = b""                
            pattern = b"|".join(re.escape(word.encode('utf-8')) for word in self.special_words)
            while p < rg[1]:
                # print(f"_pre_tokenize_from_file_place: p - rg[0] = {p - rg[0]} / {rg[1] - rg[0]}")
                text = pre_part + f.read(min(rg[1] - p, self.PRE_TOKENIZE_CHUNK_SIZE))
                p = p + min(rg[1] - p, self.PRE_TOKENIZE_CHUNK_SIZE)
                parts = re.split(pattern, text)
                pre_part = parts[-1]
                # text = re.sub(pattern, "", text)
                for part in parts[:-1]:
                    matches = re.finditer(self.PAT, part.decode('utf-8'))
                    for match in matches:
                        word = match.group().encode('utf-8')
                        words[word] = words.get(word, 0) + 1
            matches = re.finditer(self.PAT, pre_part.decode('utf-8'))
            for match in matches:
                word = match.group().encode('utf-8')
                words[word] = words.get(word, 0) + 1
        print(f"{rg[0]} to {rg[1]} completed. Time: {time.time() - st_time} s.", flush=True)
        return words
    
    def pre_tokenize_from_file(self, filename: str) -> dict[bytes, int]:
        print(f"Starting to pre-tokenize from file {filename} with parallel_num = {self.parallel_num}...")
        st_time = time.time()
        with open(filename, "rb") as f:
            chunk_boundaries = self.find_chunk_boundaries(f, self.parallel_num, b"<|endoftext|>")
            # f.seek(chunk_boundaries[1])
            # print(f.read(20))
            if len(chunk_boundaries) != self.parallel_num + 1:
                print(f"Failed to split {self.parallel_num}. Changed Parallel Num to {len(chunk_boundaries) - 1}")
                self.parallel_num = len(chunk_boundaries) - 1
        
        print("start parralel...")
        with Pool(self.parallel_num) as pool:
            input = [(filename, (chunk_boundaries[i], chunk_boundaries[i + 1])) for i in range(self.parallel_num)]
            results = pool.starmap(self._pre_tokenize_from_file_place, input)
        
        words_merge: dict[bytes, int] = results[0]
        for i in range(1, len(results)):
            d = results[i]
            new_dic = {
                key : d.get(key, 0) + words_merge.get(key, 0) for key in d.keys() | words_merge.keys()
            }
            words_merge = new_dic
        
        print(f"Done. Time elasped: {time.time() - st_time} s. Totalling {len(words_merge)} different words.")
        
        # print(words_merge.get(b"apple", 0), words_merge.get(b"I", 0), words_merge.get(b"Once", 0), words_merge.get(b"upon", 0))
        
        return words_merge
    
    def train(self, filename: str, store_file: str = None) -> None:
        
        self.train_filename = filename
        if store_file == None:
            store_file = os.path.join(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'trained_data'), os.path.basename(filename)) + ".pkl"
        self.store_filename = store_file
        
        print("Training the BPE tokenizer from start...")
        
        # initialize
        self.vocab_map = {i : bytes([i]) for i in range(256)}
        self.vocab_map.update({256 + i : self.special_words[i].encode('utf-8') for i in range(len(self.special_words))})
        self.merge_ranks = {}
        self.cur_size = 256 + len(self.special_words)
        
        # pre-tokenize
        word_dic = self.pre_tokenize_from_file(filename)
        
        # tokenization
        print("Start training...")
        st_time = time.time()
        
        self.cache_words = [WordNode(key, i, value) for i, (key, value) in enumerate(word_dic.items())]
        # self.cache_times: dict[tuple[bytes, bytes], PairNode] = defaultdict(lambda: PairNode([], [], 0, None))
        self.cache_times = {}
        
        # calc times
        for i in range(len(self.cache_words)):
            word = self.cache_words[i]
            for key, value in word.init_counts.items():
                if key not in self.cache_times:
                    if word.is_big_node:
                        self.cache_times[key] = PairNode([i], [value[1]], value[0] * word.times, key)
                    else:
                        self.cache_times[key] = PairNode([i], [None], value * word.times, key)
                else:
                    if word.is_big_node:
                        self.cache_times[key].times += value[0] * word.times
                        self.cache_times[key].in_word_nodes.append(value[1])
                        self.cache_times[key].word_positions.append(i)
                    else:
                        self.cache_times[key].times += value * word.times
                        self.cache_times[key].in_word_nodes.append(None)
                        self.cache_times[key].word_positions.append(i)

        # print(self.cache_words[:5], self.cache_words.__len__)
        # print(self.cache_times)
        
        # push into heap
        self.cache_heap = [copy(pair) for pair in self.cache_times.values()]
        heapq.heapify(self.cache_heap)
        
        mid_time = time.time()
        print(f"First round of training success. Time elasped: {mid_time - st_time} s")
            
        # BPE main loop
        while self.cur_size < self.vocab_size and self.cache_heap:
            top = self.cache_heap[0]
            heapq.heappop(self.cache_heap)
            if self.cache_times[top.byte_pair].version != top.version or top.byte_pair in self.merge_ranks: continue
            if top.times <= 0: break
            
            self.merge_ranks[top.byte_pair] = self.cur_size
            self.vocab_map[self.cur_size] = top.byte_pair[0] + top.byte_pair[1]
            # print(f"added vocab_map: {self.cur_size} to {top.byte_pair}")
            
            updates: set[tuple[bytes, bytes]] = set()
            
            for word_idx, list_nodes in zip(top.word_positions, top.in_word_nodes):
                update_now: set[tuple[bytes, bytes]]
                update_now = self.cache_words[word_idx].update(list_nodes, top.byte_pair, self.cache_times)
                updates |= update_now
                
            for tp in updates:
                self.cache_times[tp].version += 1
                heapq.heappush(self.cache_heap, copy(self.cache_times[tp]))
                
            self.cur_size += 1
            if self.cur_size % 1000 == 0:
                print(f"trained for vocab size {self.cur_size}. Time: {time.time() - mid_time} s.")
        
        ed_time = time.time()
        print(f"Training success. Time elasped from mid time: {ed_time - mid_time} s. Train time: {ed_time - st_time} s")
        self.store_to_file()
    
    def store_to_file(self, filename: str = None) -> None:
        if filename == None:
            filename = self.store_filename
        assert filename != None
        
        print(f"Storing to {filename}...")
        st_time = time.time()
        
        with open(filename, "wb") as f:
            pickle.dump(
                [self.vocab_size,
                self.cur_size,
                self.vocab_map,
                self.merge_ranks,
                self.special_words,
                self.training_backend,
                self.inference_backend,
                self.train_filename,
                self.store_filename], f
            )
        # JSON cannot represent bytes or tuple keys directly.
        # serializable_vocab_map = {
        #     str(token_id): token_bytes.hex()
        #     for token_id, token_bytes in self.vocab_map.items()
        # }
        # serializable_merge_ranks = [
        #     [left, right, rank]
        #     for (left, right), rank in self.merge_ranks.items()
        # ]
        # with open(filename, "w", encoding='utf-8') as f:
        #     json.dump(
        #         [self.vocab_size,
        #         self.cur_size,
        #         serializable_vocab_map,
        #         serializable_merge_ranks,
        #         self.special_words,
        #         self.training_backend,
        #         self.inference_backend,
        #         self.train_filename,
        #         self.store_filename],
        #         f,
        #         ensure_ascii=True,
        #         indent=2
        #     )
        
        
        print(f"Stored to file {filename}. Time: {time.time() - st_time} s.")
            
    def load_from_file(self, filename: str) -> None:
        
        print(f"Loading from {filename}...")
        st_time = time.time()
        
        with open(filename, "rb") as f:
            (self.vocab_size,
            self.cur_size,
            self.vocab_map,
            self.merge_ranks,
            self.special_words,
            self.training_backend,
            self.inference_backend,
            self.train_filename,
            self.store_filename) = pickle.load(f)
        
        # with open(filename, "r", encoding='utf-8') as f:
        #     data = json.load(f)

        # (
        #     self.vocab_size,
        #     self.cur_size,
        #     serializable_vocab_map,
        #     serializable_merge_ranks,
        #     self.special_words,
        #     self.training_backend,
        #     self.inference_backend,
        #     self.train_filename,
        #     self.store_filename,
        # ) = data

        # self.vocab_map = {
        #     int(token_id): bytes.fromhex(token_bytes)
        #     for token_id, token_bytes in serializable_vocab_map.items()
        # }
        # self.merge_ranks = {
        #     (left, right): rank
        #     for left, right, rank in serializable_merge_ranks
        # }
        
        print(f"Loaded from file {filename}. Time: {time.time() - st_time} s.")
    
    def encode_byte_bruteforce(self, word: bytes) -> list[int]:
        
        token_array = list(word)
        bytes_array = [bytes([i]) for i in word]
        while len(token_array) > 1:
            min_rank = self.vocab_size
            min_pos = -1
            for i, (b1, b2) in enumerate(zip(bytes_array[:-1], bytes_array[1:])):
                tp = (b1, b2)
                if tp in self.merge_ranks and self.merge_ranks[tp] < min_rank:
                    min_pos = i
                    min_rank = self.merge_ranks[tp]
            if min_pos == -1:
                break
            token_array = token_array[:min_pos] + [min_rank] + token_array[min_pos + 2:]
            bytes_array = bytes_array[:min_pos] + [bytes_array[min_pos] + bytes_array[min_pos + 1]] + bytes_array[min_pos + 2:]

            # print(f"bytes_array: {bytes_array}, token_array: {token_array}")
        return token_array

    class HeapNode:
        
        node: LinkedListNode[T]
        rk: int | None
        idx: int | None

        def __init__(self, node: LinkedListNode[T], rk: int = None, idx: int = None):
            self.node = node
            self.rk = rk
            self.idx = idx
        
        def __lt__(self, other: BPEtokenizer.HeapNode):
            return (self.rk, self.idx) < (other.rk, other.idx)
    
    def encode_byte_heap(self, word: bytes) -> list[int]:
        
        array = list(zip([i for i in range(len(word))], list(word), [bytes([i]) for i in word]))
        token_link = LinkedList(array, merge_op=lambda x, y: (x[0], self.merge_ranks[(x[2], y[2])], x[2] + y[2]))
        token_list = [BPEtokenizer.HeapNode(x, None, x.value[0]) for x in token_link]
        heap = []
        for i, (p1, p2) in enumerate(zip(array[:-1], array[1:])):
            tp = (p1[2], p2[2])
            if tp in self.merge_ranks:
                token_list[i].rk = self.merge_ranks[tp]
                heap.append(copy(token_list[i]))
        
        heapq.heapify(heap)
        while heap:
            now: BPEtokenizer.HeapNode = heapq.heappop(heap)
            if now.rk != token_list[now.idx].rk: continue
            assert now.rk != None
            
            nd = now.node
            token_list[nd.nxt.value[0]].rk = None
            nd.merge_with_nxt()
            
            if nd.pre != None:
                tp = (nd.pre.value[2], nd.value[2])
                pre_idx = nd.pre.value[0]
                if tp in self.merge_ranks:
                    token_list[pre_idx].rk = self.merge_ranks[tp]
                    heapq.heappush(heap, copy(token_list[pre_idx]))
                else:
                    token_list[pre_idx].rk = None
            
            token_list[now.idx].rk = None
            
            if nd.nxt != None:
                tp = (nd.value[2], nd.nxt.value[2])
                if tp in self.merge_ranks:
                    token_list[now.idx].rk = self.merge_ranks[tp]
                    heapq.heappush(heap, copy(token_list[now.idx]))
            
        return [x.value[1] for x in token_link]
        
        

    def encode(self, text: str) -> list[int]:
        if not self.special_words:
            words = self.pre_tokenize_no_special(text)
            return [x for word in words for x in self.encode_byte_bruteforce(word)]

        special_to_id = {
            word: 256 + i
            for i, word in enumerate(self.special_words)
        }
        special_pattern = "(" + "|".join(
            re.escape(word)
            for word in sorted(self.special_words, key=len, reverse=True)
        ) + ")"

        tokens = []
        for part in re.split(special_pattern, text):
            if not part:
                continue
            if part in special_to_id:
                tokens.append(special_to_id[part])
            else:
                words = self.pre_tokenize_no_special(part)
                tokens.extend(
                    x
                    for word in words
                    for x in (self.encode_byte_bruteforce(word) if len(word) < 100 else self.encode_byte_heap(word))
                )
        return tokens

    def decode(self, tokens: list[int]) -> str:
        
        return b"".join([self.vocab_map[x] for x in tokens]).decode('utf-8', errors="replace")
        # return [self.vocab_map[x] for x in tokens]
    
    def longest_tokens(self, top_k: int = 5) -> list:
        
        return sorted([i for i in self.vocab_map.items()], key=lambda x: len(x[1]), reverse = True)[:top_k]
        
    def return_adapter(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        return self.vocab_map, list(self.merge_ranks.keys())

        

"""
    python -m cProfile -s cumulative main.py # to profile your work
    
    or use 
    
    pip install scalene
    scalene main.py
"""
            
if __name__ == "__main__":
    # text = "Hello! My name is lyaaa! hhhhhh bybysfaoefjoae is good. <|endoftext|>"
    tokenizer = BPEtokenizer(vocab_size=2000, special_words=["<|endoftext|>"], parallel_num=6)
    # tokenizer = BPEtokenizer(vocab_size=32000, special_words=["<|endoftext|>"], parallel_num=6)
    store_path = "/Users/leon34/Desktop/CSdiy/stanfordCS336/Code/Lab/assignment1-basics/cs336_basics/BPETokenizer/trained_data/"
    data_path = "/Users/leon34/Desktop/CSdiy/stanfordCS336/Code/Lab/assignment1-basics/data/"
    ts_train_path = os.path.join(data_path, "TinyStoriesV2-GPT4-train.txt")
    ts_valid_path = os.path.join(data_path, "TinyStoriesV2-GPT4-valid.txt")
    owt_train_path = os.path.join(data_path, "owt_train.txt")
    owt_valid_path = os.path.join(data_path, "owt_valid.txt")
    
    ts_train_name = ""
    ts_valid_name = "_2000"
    owt_train_name = ""
    owt_valid_name = ""
    
    ts_train_store_path = os.path.join(store_path, f"ts_train{ts_train_name}.pkl")
    ts_valid_store_path = os.path.join(store_path, f"ts_valid{ts_valid_name}.pkl")
    owt_train_store_path = os.path.join(store_path, f"owt_train{owt_train_name}.pkl")
    owt_valid_store_path = os.path.join(store_path, f"owt_valid{owt_valid_name}.pkl")
    
    # tokenizer.pre_tokenize(text)
    # tokenizer.pre_tokenize_from_file(valid_path)
    # a = LinkedList(b"asdfasdfaesseefe")
    
    # for x in a:
    #     print(x.value)
        
    # b = a.head
    # b = b.nxt
    # b = b.nxt
    # b.merge_with_nxt(1)
    
    # for x in a:
    #     print(x.value)
    
    # word = b"SFDFSDF"
    # print(list(word))
    
    # tokenizer.train(ts_valid_path, store_file=ts_valid_store_path)
    # tokenizer.train(ts_train_path, store_file=ts_train_store_path)
    # tokenizer.train(owt_valid_path, store_file=owt_valid_store_path)
    # tokenizer.train(owt_train_path, store_file=owt_train_store_path)
    
    tokenizer.load_from_file(ts_valid_store_path)
    # tokenizer.load_from_file(ts_train_store_path)
    # tokenizer.load_from_file(owt_valid_store_path)
    # tokenizer.load_from_file(train_store_path)
    print(tokenizer.longest_tokens(5))
    text = "s"
    
#     text = """Once upon a time, there was a little girl named Sue. Sue was very thoughtful. She always helped her mom and dad. One day, Sue saw her mom trying to open a door with a broken handle. Sue wanted to help her mom.
# Sue asked her mom, "Can I help you?" Her mom said, "Yes, Sue. We need a new handle for the door. Can you ask dad if he has one?" Sue went to her dad and asked, "Dad, do we have a new handle for the door?" Her dad looked at Sue and said, "I am not sure, let's look together."
# Sue and her dad looked for a new handle. They found one, but it was very high up. Sue's dad tried to reach it, but he couldn't. Sue had an idea. She said, "Dad, let's use a chair to stand on." Her dad refused. He said, "No, Sue. That is not safe. Let's ask mom for help." So, they asked mom for help, and she found a safe way to get the handle. They fixed the door together, and Sue felt happy that she could help her mom and dad.
# <|endoftext|>"""
    tokens = tokenizer.encode(text)
    print(tokens)
    
    decoded = tokenizer.decode(tokens)
    print(decoded)
    
    # tokenizer2 = Tokenizer(tokenizer)
    # tokens = tokenizer.encode(text)
    # print(tokens)
    
    # decoded = tokenizer.decode(tokens)
    # print(decoded)
    # tokenizer.load_from_file("Lab1/assignment1-basics/cs336_basics/BPETokenizer/trained_data/TinyStoriesV2-GPT4-valid.txt.pkl")
