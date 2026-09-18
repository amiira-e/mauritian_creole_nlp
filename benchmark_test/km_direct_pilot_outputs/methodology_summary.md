# Pilot methodology and outputs

- Input: already reviewed workbook `pilot_gold_review_completed.xlsx`.
- Approved cases loaded: 951.
- Generators: Qwen/Qwen3-4B-Instruct-2507, Qwen/Qwen3-4B-Thinking-2507, microsoft/Phi-4-mini-instruct, HuggingFaceTB/SmolLM3-3B.
- Models receive the same case set and evidence.
- Outputs, token probabilities, log probabilities and entropy are stored.
- Rule-based checks identify formatting, abstention and unsupported identifiers.
- The automated semantic judge produces provisional labels only; it is not human gold.
- Candidate dimensions are produced by relating input/task characteristics to provisional failure categories.
- No dimension is automatically final. Promising candidates require later confirmation.
