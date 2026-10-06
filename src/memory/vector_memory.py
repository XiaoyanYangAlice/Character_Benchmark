"""
方案 B：向量记忆（RAG）。每轮对话编码存入内存向量库，当前轮检索 top-k 最相关的
历史片段拼进 prompt。

embedding 模型用 sentence-transformers（默认 intfloat/multilingual-e5-large，
中文场景效果更好；纯英文场景可以在 configs/experiment.yaml 里换成
sentence-transformers/all-MiniLM-L6-v2，体积小、SURF 上下载快）。SURF 上跑
批量编码参考 docs/SURF_GUIDE.md。

没装 sentence-transformers 时（比如本地离线开发环境），自动退化成词重叠
（Jaccard）相似度，只用来跑通整条链路的逻辑，**不能**用退化模式产出正式实验
结果——真正评测前必须确认 `_use_lexical_fallback` 是 False。

没有用 chromadb 之类的向量数据库：候选历史轮次量级（单段对话 20-40 轮）小到
没必要引入额外的存储/索引依赖，内存里暴力算余弦相似度足够快，也让代码更容易
审查（这对写评测报告、别人复现实验更重要）。
"""
import re

import numpy as np

from .base import BaseMemory

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False


# 中文没有空格分词，\w+ 会把一整句中文当成一个词，导致任何两句不同的中文
# 永远零重叠——按 Unicode CJK 区间逐字切、拉丁文字按单词切。
_TOKEN_RE = re.compile(r"[一-鿿]|[A-Za-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _lexical_similarity(query: str, text: str) -> float:
    """sentence-transformers 不可用时的退化方案：词/字集合 Jaccard 相似度。
    只是为了让流程能跑通，不代表真实的语义检索能力。"""
    q, t = _tokenize(query), _tokenize(text)
    if not q or not t:
        return 0.0
    return len(q & t) / len(q | t)


class VectorMemory(BaseMemory):
    name = "vector_memory"

    def __init__(self, embedding_model: str = "intfloat/multilingual-e5-large", top_k: int = 5):
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self._model = None
        self._use_lexical_fallback = False

        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                # ⚠ 固定用 CPU，不要让它跟着 torch 默认规则抢 GPU 0。
                # run_eval.py 跑主实验矩阵时，GPU 0/1/2 已经被三个 vllm serve
                # 子进程（qwen2.5-7b/llama3.1-8b/judge）各自吃掉几乎全部显存
                # （vllm 默认按 gpu_memory_utilization 预占），run_eval.py 这个
                # 主进程本身没有设 CUDA_VISIBLE_DEVICES，SentenceTransformer
                # 不传 device 时默认挑 cuda:0，必然 OOM。实测全量实验跑到一半
                # 发现：366 次生成里 vector_memory 这 122 次全部 OOM 失败、
                # 静默退化成词重叠相似度，且是每次实例化 VectorMemory 都重新
                # 尝试一次，不是失败一次就放弃。all-MiniLM-L6-v2 模型小、单条
                # 文本短，CPU 编码足够快，没必要跟 vLLM 抢 GPU 显存。
                self._model = SentenceTransformer(embedding_model, device="cpu")
            except Exception as e:
                print(f"[vector_memory] 加载 embedding 模型 {embedding_model} 失败（{e}），"
                      f"退化成词重叠相似度——这只适合本地调试，正式实验结果必须用真实 embedding。")
                self._use_lexical_fallback = True
        else:
            print("[vector_memory] 未安装 sentence-transformers（pip install sentence-transformers），"
                  "退化成词重叠相似度——这只适合本地调试，正式实验结果必须用真实 embedding。")
            self._use_lexical_fallback = True

        self._store: list[dict] = []  # {turn_id, speaker, text, embedding}
        # 每次 retrieve 用到了哪些历史 turn_id，供「证据可追溯率」指标计算：
        # 回复里的信息能不能定位到某条被检索出来的记忆。
        self.retrieval_log: list[dict] = []

    def _embed(self, text: str) -> np.ndarray:
        # normalize_embeddings=True 让内积直接等价于余弦相似度，不用再手动除模长
        return self._model.encode(text, normalize_embeddings=True)

    def on_new_turn(self, turn: dict) -> None:
        embedding = None if self._use_lexical_fallback else self._embed(turn["text"])
        self._store.append({**turn, "embedding": embedding})

    def retrieve(self, history: list[dict], current_turn: dict) -> str:
        if not self._store:
            self.retrieval_log.append({"query": current_turn.get("text", ""), "retrieved_turn_ids": []})
            return ""

        query_text = current_turn.get("text", "")

        if self._use_lexical_fallback:
            scored = [(_lexical_similarity(query_text, t["text"]), t) for t in self._store]
        else:
            q_emb = self._embed(query_text)
            scored = [(float(np.dot(q_emb, t["embedding"])), t) for t in self._store]

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [t for _, t in scored[: self.top_k]]
        # 检索结果按原始时间顺序排回去再拼进 prompt，读起来更像连贯的对话片段，
        # 不打乱模型对时间先后的理解（相似度排序只用来选 topk，不用来定呈现顺序）。
        top.sort(key=lambda t: t.get("turn_id", 0))

        self.retrieval_log.append({
            "query": query_text,
            "retrieved_turn_ids": [t.get("turn_id") for t in top],
        })

        lines = [f"{t['speaker']}: {t['text']}" for t in top]
        return "\n".join(lines)
