# Recipes · Рецепты

Starting points for `Qwen/Qwen2.5-0.5B-Instruct` (24 layers). Load one with **Импорт рецепта / Import recipe**, then tweak the strengths in the medical chart — every model reacts differently.

Стартовые рецепты для `Qwen/Qwen2.5-0.5B-Instruct` (24 слоя). Загрузите через **«Импорт рецепта»** и подкрутите силу под себя — каждая модель реагирует по-своему.

| Recipe | What it does | Ask it |
|---|---|---|
| `drunk-philosopher.json` | Layer volume ×2.5 on layers 10–12 | "What is the meaning of life?" |
| `amnesia.json` | No MLP in layers 4–12 | "What is the capital of France?" |
| `catdog.json` | Swaps *cat* ↔ *dog* in the vocabulary | "Tell me about dogs." |
| `depressed-pirate.json` | Pirate mood in layer 11, inverted joy in layer 12 | "How was your day?" |
| `groundhog-day.json` | Layer 16 runs 7 times in a row | "Count from 1 to 10." |
| `backwards-brain.json` | Layers 8–15 run in reverse order | "Write a short poem about autumn." |
| `fever-dream.json` | MLP noise in layers 6–17 plus an inverted layer 14 | "Describe your ideal day." |

Got a better one? Open a PR with your JSON and a sample answer.
