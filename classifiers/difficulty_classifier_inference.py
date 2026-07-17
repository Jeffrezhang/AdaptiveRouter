"""
Difficulty classifier inference module.

Loads Qwen3-0.6B + MS-Swift LoRA adapter and classifies prompts
as easy / medium / hard via text generation.

Usage as a module:
    from difficulty_classifier_inference import DifficultyClassifier
    clf = DifficultyClassifier()
    clf.classify("What is the capital of France?", "Open QA")  # -> "easy"

Usage from command line:
    python difficulty_classifier_inference.py "Implement dynamic programming" "Coding"
"""

import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(_BASE_DIR, "difficulty-classifier-final")
MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_LEN = 256

SYSTEM_PROMPT = """Classify the difficulty of the following prompt as: easy, medium, or hard.

Rules:
- Coding prompts are medium by default. Escalate to hard for: dynamic programming, graph algorithms, distributed systems, concurrency/thread-safety, cryptography, ML model internals, system design at scale, or implementing complex data structures from scratch.
- Closed QA prompts are medium by default. Escalate to hard for mathematical proofs, advanced theory, or multi-step derivations.
- Open QA, Generation, Brainstorm prompts are easy UNLESS the topic requires deep domain expertise.
- Chat, Rewrite, Summarize, Extract, Classify prompts are easy UNLESS unusually complex.
- The category gives a baseline difficulty the prompt cannot lower.

hard = dynamic programming, graph algorithms, distributed systems, concurrency, cryptography, ML internals, system design at scale, complex data structures from scratch, mathematical proofs
medium = general coding tasks, debugging, SQL, API design, structured writing, comparisons, explanations
easy = factual questions, casual chat, simple creative writing, basic how-to questions

Output ONLY one word: easy, medium, or hard."""


class DifficultyClassifier:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        adapter_path: str = ADAPTER_PATH,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading difficulty classifier on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.to(self.device)
        self.model.eval()
        print("Difficulty classifier ready!")

    def classify(self, query: str, category: str) -> str:
        """
        Classify a prompt's difficulty.

        Args:
            query: The user prompt text
            category: The prompt category (e.g. 'Coding', 'Open QA')

        Returns:
            'easy', 'medium', or 'hard'
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Category: {category}\nPrompt: {query}"},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip().lower()

        match = re.search(r"\b(easy|medium|hard)\b", response)
        return match.group(1) if match else "easy"

    def classify_batch(self, queries: list) -> list:
        """
        Classify a batch of (query, category) tuples.

        Args:
            queries: List of (query, category) tuples

        Returns:
            List of difficulty labels
        """
        return [self.classify(q, c) for q, c in queries]


if __name__ == "__main__":
    clf = DifficultyClassifier()

    if len(sys.argv) >= 3:
        query = sys.argv[1]
        category = sys.argv[2]
        result = clf.classify(query, category)
        print(f"Difficulty: {result}")
    else:
        # Run default test prompts
        test_prompts = [
            ("Open QA", "What is the capital of France?"),
            ("Coding", "Write a Python function to reverse a string"),
            ("Brainstorm", "Give me YouTube video ideas about AI"),
            ("Coding", "Implement a distributed consensus algorithm using Raft"),
            ("Coding", "How do I implement dynamic programming for the knapsack problem?"),
        ]
        for category, query in test_prompts:
            result = clf.classify(query, category)
            print(f"[{result}] [{category}] {query}")