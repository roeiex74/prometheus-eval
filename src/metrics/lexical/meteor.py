"""
METEOR (Metric for Evaluation of Translation with Explicit ORdering) implementation.
Based on Banerjee & Lavie (2005) - combines precision, recall, and alignment-based
fragmentation penalty. Uses exact, stem, and synonym matching stages.
"""
from typing import List, Dict, Union, Tuple, Set
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
import nltk

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


class METEORMetric:
    """METEOR metric with alignment-based matching and fragmentation penalty."""

    def __init__(self, alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5):
        """Initialize METEOR with F-measure weights and penalty parameter.

        Args:
            alpha: Recall weight in F-mean (default 0.9 for 9:1 recall:precision)
            beta: Penalty exponent (default 3.0)
            gamma: Penalty coefficient (default 0.5)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.stemmer = PorterStemmer()

    def compute(self, candidate: str, references: Union[str, List[str]], **kwargs) -> Dict[str, float]:
        """Compute METEOR score using exact, stem, and synonym matching.
        Returns dict with: 'meteor', 'precision', 'recall', 'f_mean', 'penalty', 'chunks'
        """
        if isinstance(references, str):
            references = [references]
        best_score = None
        for ref in references:
            score = self._compute_single(candidate, ref)
            if best_score is None or score['meteor'] > best_score['meteor']:
                best_score = score
        return best_score if best_score else self._empty_score()

    def _compute_single(self, candidate: str, reference: str) -> Dict[str, float]:
        """Compute METEOR for a single candidate-reference pair."""
        cand_tokens, ref_tokens = candidate.lower().split(), reference.lower().split()
        if not cand_tokens or not ref_tokens:
            return self._empty_score()
        alignments = self._align_words(cand_tokens, ref_tokens)
        return self._compute_score(len(alignments), len(cand_tokens), len(ref_tokens), self._count_chunks(alignments))

    def _align_words(self, cand_tokens: List[str], ref_tokens: List[str]) -> List[Tuple[int, int]]:
        """Create greedy alignment: exact -> stem -> synonym. Each ref token matched once."""
        cand_stems, ref_stems = self._get_stems(cand_tokens), self._get_stems(ref_tokens)
        used_ref, alignments = set(), []
        # Stage 1: Exact match
        for i, cw in enumerate(cand_tokens):
            for j, rw in enumerate(ref_tokens):
                if j not in used_ref and cw == rw:
                    alignments.append((i, j))
                    used_ref.add(j)
                    break
        # Stage 2: Stem match
        for i, cs in enumerate(cand_stems):
            if any(a[0] == i for a in alignments):
                continue
            for j, rs in enumerate(ref_stems):
                if j not in used_ref and cs == rs:
                    alignments.append((i, j))
                    used_ref.add(j)
                    break
        # Stage 3: Synonym match
        for i, cw in enumerate(cand_tokens):
            if any(a[0] == i for a in alignments):
                continue
            cand_syns = self._get_synonyms(cw)
            for j, rw in enumerate(ref_tokens):
                if j not in used_ref and cand_syns & self._get_synonyms(rw):
                    alignments.append((i, j))
                    used_ref.add(j)
                    break
        return sorted(alignments)

    def _get_stems(self, tokens: List[str]) -> List[str]:
        """Get Porter stems for tokens."""
        return [self.stemmer.stem(t) for t in tokens]

    def _get_synonyms(self, word: str) -> Set[str]:
        """Get WordNet synonyms for a word."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower())
        return synonyms

    def _count_chunks(self, alignments: List[Tuple[int, int]]) -> int:
        """Count consecutive alignment chunks (new chunk if either index not consecutive)."""
        if not alignments:
            return 0
        chunks = 1
        for i in range(1, len(alignments)):
            prev_c, prev_r = alignments[i - 1]
            curr_c, curr_r = alignments[i]
            if curr_c != prev_c + 1 or curr_r != prev_r + 1:
                chunks += 1
        return chunks

    def _compute_score(self, matches: int, cand_len: int, ref_len: int, chunks: int) -> Dict[str, float]:
        """Compute METEOR = (1 - Penalty) × F_mean where F_mean = (10PR)/(R+9P)."""
        if matches == 0:
            return self._empty_score()
        precision = matches / cand_len if cand_len > 0 else 0.0
        recall = matches / ref_len if ref_len > 0 else 0.0
        f_mean = (10 * precision * recall) / (recall + 9 * precision) if (precision + recall) > 0 else 0.0
        penalty = self.gamma * ((chunks / matches) ** self.beta)
        meteor = (1 - penalty) * f_mean
        return {'meteor': meteor, 'precision': precision, 'recall': recall,
                'f_mean': f_mean, 'penalty': penalty, 'chunks': chunks}

    def _empty_score(self) -> Dict[str, float]:
        """Return zero score for empty inputs."""
        return {'meteor': 0.0, 'precision': 0.0, 'recall': 0.0, 'f_mean': 0.0, 'penalty': 0.0, 'chunks': 0}
