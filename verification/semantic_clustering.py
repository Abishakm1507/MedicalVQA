"""Semantic equivalence clustering and normalized semantic entropy.

Clusters N sampled VLM answers into semantic equivalence classes and computes
agreement (a) and normalized semantic entropy (H_n) per Kuhn et al. (2023)
and ADR 0005. Uses an embedding-based approach as the primary backend with a
deterministic canonical-entity fallback when no embedding model is available.

Traceability:
  - Feature Specification: FS-005 (Phase 2), section 4.5
  - ADR 0005: Confidence Gate Formulation, Semantic Clustering & Calibration
  - Research Paper: Section III-E (Confidence Gate)
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class SemanticCluster:
    """A cluster of semantically equivalent sampled answers."""

    cluster_id: int
    members: list[int]  # indices of sampled answers in this cluster
    representative: str  # canonical answer text of the cluster


@dataclass
class AgreementEntropyResult:
    """Result of semantic agreement and entropy computation.

    ``is_valid`` is True only when the sampling produced a usable result
    (n_samples > 0 and no error). Invalid results (empty/missing/malformed
    sampling) have is_valid=False and must NOT be fed into confidence
    calculation — the runtime gate must abstain instead.
    """

    majority_answer: str
    agreement: float  # a = max_k |C_k| / N, in [0, 1]
    raw_entropy: float  # H = -sum p_k ln p_k, in [0, ln(N)]
    normalized_entropy: float  # H_n = H / ln(N), in [0, 1]
    clusters: list[SemanticCluster]
    n_samples: int
    clustering_method: Literal["embedding", "canonical_fallback"]
    error: str | None = None
    is_valid: bool = False

    def __post_init__(self) -> None:
        """Derive is_valid from n_samples and error."""
        self.is_valid = self.n_samples > 0 and self.error is None


class SemanticClusterer:
    """Clusters sampled answers into semantic equivalence classes.

    Primary backend: sentence-transformer embeddings with cosine-similarity
    threshold clustering (recommended model: all-MiniLM-L6-v2).
    Fallback backend: canonical entity normalization (lowercase, strip,
    collapse whitespace) with exact-match clustering.

    The cosine threshold is a configurable parameter, not a scientifically
    validated constant. It must be calibrated on held-out validation data
    before benchmark use (per ADR 0005).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        force_fallback: bool = False,
    ) -> None:
        """Initialize the clusterer.

        Args:
            similarity_threshold: Cosine similarity threshold for merging
                answers into the same cluster. Configurable; calibrated on
                validation data before benchmark use. Not a validated
                scientific constant.
            embedding_model_name: Sentence-Transformer model name (only used
                if sentence-transformers is installed and force_fallback=False).
            force_fallback: If True, skip the embedding backend and use the
                canonical-fallback clustering even if embeddings are available.
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model_name
        self.force_fallback = force_fallback
        self._embedding_backend = None
        self._backend_name: Literal["embedding", "canonical_fallback"] = (
            "canonical_fallback" if force_fallback else "embedding"
        )
        self._init_backend()

    def _init_backend(self) -> None:
        """Attempt to initialize the embedding backend.

        Falls back to canonical clustering if sentence-transformers is not
        installed or initialization fails. No exception is raised.
        """
        if self.force_fallback:
            self._backend_name = "canonical_fallback"
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer(self.embedding_model_name)
            self._embedding_backend = model
            self._backend_name = "embedding"
        except Exception:
            self._embedding_backend = None
            self._backend_name = "canonical_fallback"

    @property
    def backend_name(self) -> Literal["embedding", "canonical_fallback"]:
        return self._backend_name

    def _canonical_normalize(self, text: str) -> str:
        """Canonical normalization for fallback clustering.

        Lowercases, strips, and collapses whitespace. Does NOT remove
        punctuation or perform medical entity normalization (UMLS/SNOMED
        would be required for that and is a separate integration concern).
        """
        return " ".join(text.strip().lower().split())

    def _compute_similarity_matrix(self, answers: list[str]) -> list[list[float]]:
        """Compute pairwise cosine similarity matrix.

        If the embedding backend is available, uses sentence embeddings.
        Otherwise, uses exact-match on canonical normalization (1.0 if
        canonical forms match, 0.0 otherwise).
        """
        n = len(answers)
        matrix = [[0.0] * n for _ in range(n)]
        if n == 0:
            return matrix

        if self._embedding_backend is not None:
            try:
                import numpy as np  # type: ignore

                embeddings = self._embedding_backend.encode(
                    answers, convert_to_numpy=True, normalize_embeddings=True
                )
                for i in range(n):
                    for j in range(n):
                        matrix[i][j] = float(
                            np.dot(embeddings[i], embeddings[j])
                        )
                return matrix
            except Exception:
                # Fall through to canonical fallback on any embedding error.
                pass

        # Canonical fallback: exact match on normalized form.
        canonical = [self._canonical_normalize(a) for a in answers]
        for i in range(n):
            for j in range(n):
                matrix[i][j] = 1.0 if canonical[i] == canonical[j] else 0.0
        return matrix

    def cluster(self, answers: list[str]) -> list[SemanticCluster]:
        """Cluster answers into semantic equivalence classes.

        Uses greedy threshold-based connected-component clustering: two answers
        are in the same cluster if their similarity >= threshold, transitively
        closed via chain clustering.

        TRANSITIVITY NOTE: Connected-component clustering is transitive — if
        answer A is similar to B and B is similar to C, then A, B, and C are
        placed in the same cluster even if A is not directly similar to C.
        This can over-merge answers in chains of near-matches. For medical VQA
        answers this is generally acceptable because clinical paraphrases tend
        to cluster tightly, but users should be aware that the clustering is
        not purely pairwise-equivalence based. If strict pairwise equivalence
        is required, replace this with a different clustering algorithm
        (e.g. complete-linkage hierarchical clustering).

        Args:
            answers: List of N sampled answer strings.

        Returns:
            List of SemanticCluster objects.
        """
        n = len(answers)
        if n == 0:
            return []

        sim = self._compute_similarity_matrix(answers)
        visited = [False] * n
        clusters: list[SemanticCluster] = []
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            # BFS/DFS to find all answers connected to i via threshold.
            stack = [i]
            visited[i] = True
            members: list[int] = []
            while stack:
                cur = stack.pop()
                members.append(cur)
                for j in range(n):
                    if not visited[j] and sim[cur][j] >= self.similarity_threshold:
                        visited[j] = True
                        stack.append(j)
            members.sort()
            representative = answers[members[0]]
            clusters.append(
                SemanticCluster(
                    cluster_id=cluster_id,
                    members=members,
                    representative=representative,
                )
            )
            cluster_id += 1

        return clusters

    def compute_agreement_entropy(
        self, answers: list[str]
    ) -> AgreementEntropyResult:
        """Compute agreement (a) and normalized semantic entropy (H_n).

        Args:
            answers: List of N sampled answer strings.

        Returns:
            AgreementEntropyResult with agreement, entropy, and cluster info.
        """
        n = len(answers)
        if n == 0:
            return AgreementEntropyResult(
                majority_answer="",
                agreement=0.0,
                raw_entropy=0.0,
                normalized_entropy=0.0,
                clusters=[],
                n_samples=0,
                clustering_method=self._backend_name,
                error="no_samples",
            )

        clusters = self.cluster(answers)

        # Cluster probabilities p_k = |C_k| / N.
        probs = [len(c.members) / n for c in clusters]

        # Agreement: modal cluster proportion.
        max_prob = max(probs)
        modal_idx = int(probs.index(max_prob))
        majority_answer = clusters[modal_idx].representative

        # Raw entropy: H = -sum p_k ln p_k (0 ln 0 = 0).
        import math

        raw_entropy = 0.0
        for p in probs:
            if p > 0.0:
                raw_entropy -= p * math.log(p)

        # Normalized entropy: H_n = H / ln(N).
        if n == 1:
            normalized_entropy = 0.0
        else:
            normalized_entropy = raw_entropy / math.log(n)

        return AgreementEntropyResult(
            majority_answer=majority_answer,
            agreement=max_prob,
            raw_entropy=raw_entropy,
            normalized_entropy=normalized_entropy,
            clusters=clusters,
            n_samples=n,
            clustering_method=self._backend_name,
            error=None,
        )
