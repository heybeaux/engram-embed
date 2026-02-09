//! Test fixtures for engram-embed tests

/// Short text sample (simple sentence)
pub const SHORT_TEXT: &str = "The quick brown fox jumps over the lazy dog.";

/// Long text sample (>512 tokens to test truncation)
/// This is a repeated passage to ensure we exceed the token limit
pub const LONG_TEXT: &str = r#"
Artificial intelligence (AI) is transforming the way we live and work. From machine learning algorithms that power recommendation systems to natural language processing models that understand and generate human text, AI technologies are becoming increasingly sophisticated and ubiquitous.

The field of deep learning has seen remarkable advances in recent years. Neural networks with millions or even billions of parameters can now perform tasks that were once thought to be exclusively human capabilities. Image recognition systems can identify objects with superhuman accuracy. Language models can write coherent essays, translate between languages, and answer complex questions.

Embedding models, like the ones tested here, are fundamental building blocks of modern NLP systems. They convert text into dense vector representations that capture semantic meaning. Similar texts produce similar vectors, enabling powerful applications like semantic search, clustering, and classification.

The transformer architecture, introduced in the landmark paper "Attention Is All You Need," revolutionized the field. Self-attention mechanisms allow these models to capture long-range dependencies in text, leading to dramatic improvements in performance across virtually all NLP tasks.

Training these models requires massive computational resources and enormous datasets. Companies and research institutions have invested billions of dollars in building ever-larger models. The scaling laws suggest that model capabilities continue to improve with size, though there are ongoing debates about the environmental and economic sustainability of this approach.

Despite the impressive capabilities of modern AI systems, there are important limitations and challenges. Models can exhibit biases present in their training data. They can confidently produce incorrect information. They lack true understanding and reasoning capabilities in the way humans possess them.

Safety and alignment research has become increasingly important as AI systems become more capable. Ensuring that AI systems behave in accordance with human values and intentions is one of the great challenges of our time. Researchers are developing new techniques for interpretability, robustness, and value alignment.

The economic implications of AI are profound. Automation threatens to displace many jobs, while simultaneously creating new opportunities. The need for AI literacy and new skills is becoming apparent across industries.

Regulatory frameworks are being developed to govern the use of AI. Questions about liability, privacy, fairness, and transparency are at the forefront of policy discussions. Different jurisdictions are taking varied approaches to AI governance.

Looking ahead, the trajectory of AI development remains uncertain but exciting. Advances in areas like multimodal learning, reasoning, and embodied AI promise new capabilities. The integration of AI into robotics, healthcare, education, and other domains will continue to accelerate.

The democratization of AI through open-source models and accessible APIs is enabling a new generation of developers and entrepreneurs to build innovative applications. Tools that once required specialized expertise are now available to anyone with an internet connection.

Research continues at a rapid pace. Academic institutions and industry labs are publishing thousands of papers each year, pushing the boundaries of what's possible. The cross-pollination of ideas between academia and industry has created a vibrant ecosystem of innovation.
"#;

/// Unicode/multilingual text sample
pub const UNICODE_TEXT: &str = r#"
Hello, World! 你好世界！Привет мир! مرحبا بالعالم 🌍🌎🌏

日本語のテキストもサポートされています。機械学習モデルは多言語テキストを処理できます。

Emoji are also supported: 🚀 💡 🤖 📚 ✨ 🎯 💻 🔬

Mathematical symbols: ∑ ∏ ∫ √ ∞ ≈ ≠ ≤ ≥ ∈ ∉ ⊂ ⊃

Greek letters: α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω

Special characters: € £ ¥ © ® ™ § ¶ † ‡ • ◦ ‣ ° ′ ″

Combining characters: é è ê ë ē ě ę ë ə

Arabic text: مرحبا بالعالم العربي، نحن نختبر نموذج التضمين.

Korean text: 안녕하세요! 임베딩 모델을 테스트하고 있습니다.

Thai text: สวัสดีครับ ทดสอบระบบฝังข้อความ

Hebrew text: שלום עולם! זהו טקסט בדיקה.
"#;

/// Code sample (programming language text)
pub const CODE_TEXT: &str = r#"
```rust
use std::collections::HashMap;
use anyhow::Result;

/// A simple cache implementation
pub struct Cache<K, V> {
    data: HashMap<K, V>,
    max_size: usize,
}

impl<K: Eq + std::hash::Hash, V> Cache<K, V> {
    /// Create a new cache with the given maximum size
    pub fn new(max_size: usize) -> Self {
        Self {
            data: HashMap::new(),
            max_size,
        }
    }

    /// Insert a key-value pair into the cache
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.data.len() >= self.max_size && !self.data.contains_key(&key) {
            return None; // Cache is full
        }
        self.data.insert(key, value)
    }

    /// Get a value from the cache
    pub fn get(&self, key: &K) -> Option<&V> {
        self.data.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = Cache::new(10);
        cache.insert("key1", "value1");
        assert_eq!(cache.get(&"key1"), Some(&"value1"));
    }
}
```

```python
import numpy as np
from typing import List, Optional

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

class EmbeddingIndex:
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.texts: List[str] = []
    
    def add(self, text: str, embedding: np.ndarray):
        self.texts.append(text)
        self.embeddings.append(embedding)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[tuple]:
        scores = [cosine_similarity(query_embedding, e) for e in self.embeddings]
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self.texts[i], scores[i]) for i in indices]
```
"#;

/// Empty string for edge case testing
pub const EMPTY_TEXT: &str = "";

/// Single character text
pub const SINGLE_CHAR_TEXT: &str = "a";

/// Whitespace-only text
pub const WHITESPACE_TEXT: &str = "   \t\n   ";

/// Very long repeated word (to test tokenization edge cases)
pub fn very_long_word() -> String {
    "supercalifragilisticexpialidocious".repeat(100)
}

/// Generate a text with exactly n approximate tokens (rough estimate)
pub fn text_with_approx_tokens(n: usize) -> String {
    let base_word = "word ";
    base_word.repeat(n)
}
