from sentence_transformers import SentenceTransformer
import torch

class ProgramEmbedder:
    def __init__(self, ckpt: str = "Salesforce/codet5p-110m-embedding"):
        self.model = SentenceTransformer(ckpt, device="cuda")

    @torch.no_grad()
    def encode(self, code_str: str) -> torch.Tensor:
        """Return 128-dim normalized embedding for provided code."""
        v = self.model.encode([code_str], normalize_embeddings=True)[0]
        return torch.tensor(v, dtype=torch.float16)[:128]

if __name__ == "__main__":
    emb = ProgramEmbedder()
    vec = emb.encode("def add(a,b): return a+b")
    print(vec.reshape(1, -1))
