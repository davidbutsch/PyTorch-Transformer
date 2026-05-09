import regex
import unicodedata
from tqdm import tqdm

from config import get_merges_path, get_vocabs_path


class RegexTokenizer:

    def __init__(self) -> None:
        # We only consider byte pairs in chunks defined by the split pattern
        self.pattern = None

        # Binary forest (built bottom-up)
        # { pair: token_id }
        self.merges: dict[tuple[int, int], int] = {}

        # Maps token_id to utf-8 bytes
        self.vocabs: dict[int, bytes] = {}

        self.special_tokens: dict[str, int] | None = None
        self.inverse_special_tokens: dict[int, str] | None = None

    # Get the most frequent byte pair from chunked list of token ids
    def _get_top_pair_chunked(
        self, chunked_token_ids: list[list[int]]
    ) -> tuple[int, int] | None:
        # Keep track of pair frequency across all chunks
        # { pair: total_frequency }
        total_freq = {}

        for chunk in chunked_token_ids:
            # If chunk contains no pairs, continue to next chunk
            if len(chunk) < 2:
                continue

            # Get frequencies dict for this chunk
            chunk_freq = self._get_byte_pair_frequencies(chunk)

            # Build total_freq from chunk frequencies
            for pair_k, freq_v in chunk_freq.items():
                total_freq[pair_k] = total_freq.get(pair_k, 0) + freq_v

        # If total_freq is empty, return None
        if len(total_freq) == 0:
            return None

        # Otherwise return the total top pair
        top_pair = max(total_freq, key=total_freq.get)  # type: ignore
        return top_pair

    # Get the most frequent byte pair from flat list of token ids
    def _get_top_pair(self, token_ids: list[int]):
        # If token list contains no pairs, return None
        if len(token_ids) < 2:
            return None

        # Get frequencies for each byte pair
        freq = self._get_byte_pair_frequencies(token_ids)

        # Return most frequent byte pair
        top_pair = max(freq, key=freq.get)  # type: ignore
        return top_pair

    # Get byte pair frequencies dict
    def _get_byte_pair_frequencies(
        self, token_ids: list[int]
    ) -> dict[tuple[int, int], int]:
        # { pair: frequency }
        freq = {}

        # Sliding window tuple
        for pair in zip(token_ids, token_ids[1:]):
            freq[pair] = freq.get(pair, 0) + 1

        return freq

    def _merge_chunked(self, chunked_token_ids: list[list[int]], pair, new_token_id):
        chunked_merged = []

        # Merge each chunk
        for chunk in chunked_token_ids:
            chunked_merged.append(self._merge(chunk, pair, new_token_id))

        return chunked_merged

    def _merge(self, token_ids: list[int], pair, new_token_id) -> list[int]:
        merged = []

        i = 0
        while i < len(token_ids):
            # If pair match is found
            if (
                i + 1 < len(token_ids)
                and token_ids[i] == pair[0]
                and token_ids[i + 1] == pair[1]
            ):
                # Append the new_token_id and skip ahead two slots
                merged.append(new_token_id)
                i += 2
            else:
                # Otherwise simply insert the current token and move on
                merged.append(token_ids[i])
                i += 1

        return merged

    def _build_vocabs(self):

        # Initialize vocabs with ascii bytes
        vocabs = {token_id: bytes([token_id]) for token_id in range(256)}

        for (tok0, tok1), token_id in self.merges.items():
            # Inorder traversal means that self.vocabs[tok{i}] is always defined
            vocabs[token_id] = vocabs[tok0] + vocabs[tok1]

        return vocabs

    def train(self, text, vocab_size):
        assert self.pattern and self.special_tokens is not None

        # e.g. ['I', ' think', ',', ' therefore', ' I', ' am']
        chunks: list[str] = regex.findall(self.pattern, text)

        # e.g. [[67, 111, 112, 121], [32, 111, 102], ..., [32, 116, 104, 101]]
        # We initialize the token_ids as unicode utf-8 bytes
        # Frequent byte pairs are merged into new tokens with token_id >= 256
        chunked_token_ids: list[list[int]] = [
            list(chunk.encode("utf-8")) for chunk in chunks
        ]

        # Compute number of merges to perform such that we reach `vocab_size`
        num_merges = vocab_size - 256 - len(self.special_tokens)

        # Perform merges
        for i in tqdm(range(num_merges), desc="merges"):
            # Compute the top byte pair for this merge round
            top_pair = self._get_top_pair_chunked(chunked_token_ids)
            new_token_id = 256 + i

            # If no more pairs, we done
            if not top_pair:
                break

            # Merge the top pair
            chunked_token_ids = self._merge_chunked(
                chunked_token_ids, top_pair, new_token_id
            )

            # Save this merge operation to merges
            self.merges[top_pair] = new_token_id

            tqdm.write(f"Merge {i+1}: {top_pair} -> {new_token_id}")

        # Build vocabs mapping after training
        self.vocabs = self._build_vocabs()

    def register_special_tokens(self, special_tokens: dict[str, int]):
        # e.g. { "<|endoftext|>": 100257 }
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def register_pattern(self, pattern: str):
        self.pattern = regex.compile(pattern)

    def save(self):
        """
        Saves two files: prefix.merges, and prefix.vocabs
        - prefix.merges is the important one, intended for load()
        - prefix.vocabs is just a pretty printed version for human inspection

        *copied pretty much directly from karpathy/minbpe
        """

        # Save merge file
        merge_file = get_merges_path()

        with open(merge_file, "w") as f:
            # Write the pattern used
            f.write(f"{self.pattern}\n")

            # Write the merges dict
            for tok0, tok1 in self.merges:
                f.write(f"{tok0} {tok1}\n")

        # Save vocabs file
        vocabs_file = get_vocabs_path()

        def render_token(token: bytes) -> str:
            # Pretty print token bytes
            s = token.decode("utf-8", errors="replace")

            # Replace control characters
            chars = []
            for ch in s:
                if unicodedata.category(ch)[0] != "C":
                    chars.append(ch)  # this character is ok
                else:
                    chars.append(f"\\u{ord(ch):04x}")  # escape
            s = "".join(chars)

            return s

        with open(vocabs_file, "w", encoding="utf-8") as f:
            inverted_merges = {token_id: pair for pair, token_id in self.merges.items()}

            for token_id, token_bytes in self.vocabs.items():
                s = render_token(token_bytes)

                # Display token children as merge
                if token_id in inverted_merges:
                    tok0, tok1 = inverted_merges[token_id]
                    s0 = render_token(self.vocabs[tok0])
                    s1 = render_token(self.vocabs[tok1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {token_id}\n")

                else:
                    # Otherwise this is a leaf node, just print it
                    f.write(f"[{s}] {token_id}\n")

    def load(self):
        """
        Loads the merges file
        """

        merges_file = get_merges_path()
        cur_token_id = 256
        with open(merges_file, "r", encoding="utf-8") as f:
            # Read in pattern
            self.pattern = f.readline().strip()

            # Read in merges
            for line in f:
                id1, id2 = map(int, line.split())
                self.merges[(id1, id2)] = cur_token_id
                cur_token_id += 1

        self.vocabs = self._build_vocabs()

    # Encode text to token_ids
    def _encode_ordinary(self, text: str):
        token_ids = list(text.encode("utf-8"))

        while len(token_ids) >= 2:
            bp_freq = self._get_byte_pair_frequencies(token_ids)

            # Merge bottom-up to respect dependency chain (higher token_ids are composed of lower token_ids)
            pair = min(bp_freq, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break  # Nothing more to merge

            # Perform merge
            new_token_id = self.merges[pair]
            token_ids = self._merge(token_ids, pair, new_token_id)

        return token_ids

    def encode(self, text: str, allowed_special="none_raise") -> list[int]:
        """
        Unlike _encode_ordinary, this function handles special tokens.
        allowed_special can be:
        - "all" -> all special tokens acceptable
        - "none" -> no special tokens acceptable
        - "none_raise" -> error is raised if encountering any special tokens
        - custom set of special tokens
        """
        assert self.special_tokens is not None

        current_special: dict[str, int] = {}

        if allowed_special == "all":
            current_special = self.special_tokens
        elif allowed_special == "none":
            current_special = {}
        elif allowed_special == "none_raise":
            current_special = {}
            # Verify that no special token exists in text
            assert all(k not in text for k in self.special_tokens)
        elif isinstance(allowed_special, set):
            # Build special dict from allowed_special string set
            current_special = {
                k: v for k, v in self.special_tokens.items() if k in allowed_special
            }
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        # If no currently acceptable special tokens, process normally
        if not current_special:
            return self._encode_ordinary(text)

        # Otherwise, handle currently acceptable special tokens

        # Catch special tokens and chunk the text
        special_pattern = (
            "(" + "|".join(regex.escape(k) for k in self.special_tokens) + ")"
        )
        chunks = regex.split(special_pattern, text)

        token_ids: list[int] = []

        # Process each chunk
        for chunk in chunks:
            # If chunk is a special token, pull from current special tokens dict
            if chunk in current_special:
                token_ids.append(current_special[chunk])
            else:
                token_ids.extend(self._encode_ordinary(chunk))

        return token_ids

    # Decode token_ids to text bytes
    def decode(self, token_ids: list[int], separator=b"") -> str:
        assert self.inverse_special_tokens is not None

        token_bytes: list[bytes] = []

        for token_id in token_ids:
            if token_id in self.vocabs:
                # If regular token, decode with vocabs mapping
                token_bytes.append(self.vocabs[token_id])
            elif token_id in self.inverse_special_tokens:
                # If special token, decode with special tokens mapping
                token_bytes.append(
                    self.inverse_special_tokens[token_id].encode("utf-8")
                )
            else:
                raise ValueError(f"Invalid token id: {token_id}")

        text_bytes = separator.join(token_bytes)

        # Errors="replace" replaces invalid bytes with the unicode replacement character, �
        text = text_bytes.decode("utf-8", errors="replace")
        return text
