//! Simple tokenizer for MDR (placeholder - use sentencepiece/tiktoken in production)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Simple character-level tokenizer for testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleTokenizer {
    pub char_to_id: HashMap<char, usize>,
    pub id_to_char: HashMap<usize, char>,
    pub vocab_size: usize,
}

impl SimpleTokenizer {
    /// Create a basic ASCII tokenizer
    pub fn ascii() -> Self {
        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();

        // Special tokens
        char_to_id.insert('\0', 0);  // PAD
        id_to_char.insert(0, '\0');

        char_to_id.insert('\x01', 1);  // BOS
        id_to_char.insert(1, '\x01');

        char_to_id.insert('\x02', 2);  // EOS
        id_to_char.insert(2, '\x02');

        char_to_id.insert('\x03', 3);  // UNK
        id_to_char.insert(3, '\x03');

        // Printable ASCII (32-126)
        for (i, c) in (32u8..=126).enumerate() {
            let c = c as char;
            let id = i + 4;
            char_to_id.insert(c, id);
            id_to_char.insert(id, c);
        }

        // Newline
        let newline_id = 4 + 95;
        char_to_id.insert('\n', newline_id);
        id_to_char.insert(newline_id, '\n');

        Self {
            vocab_size: newline_id + 1,
            char_to_id,
            id_to_char,
        }
    }

    /// Encode string to token IDs
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = vec![1];  // BOS
        for c in text.chars() {
            let id = self.char_to_id.get(&c).copied().unwrap_or(3);  // UNK
            tokens.push(id);
        }
        tokens.push(2);  // EOS
        tokens
    }

    /// Decode token IDs to string
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter_map(|&id| self.id_to_char.get(&id))
            .filter(|&&c| c >= ' ' || c == '\n')  // Skip control chars except newline
            .collect()
    }

    /// BOS token ID
    pub fn bos_id(&self) -> usize {
        1
    }

    /// EOS token ID
    pub fn eos_id(&self) -> usize {
        2
    }

    /// PAD token ID
    pub fn pad_id(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() {
        let tok = SimpleTokenizer::ascii();

        let text = "Hello, World!";
        let tokens = tok.encode(text);

        assert_eq!(tokens[0], 1);  // BOS
        assert_eq!(*tokens.last().unwrap(), 2);  // EOS

        let decoded = tok.decode(&tokens);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_unknown_char() {
        let tok = SimpleTokenizer::ascii();

        let text = "Hello 你好";  // Chinese chars not in ASCII
        let tokens = tok.encode(text);

        // Should contain UNK tokens for Chinese chars
        assert!(tokens.contains(&3));
    }
}
