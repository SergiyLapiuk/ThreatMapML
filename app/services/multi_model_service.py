from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
import time

from app.services.utils import extract_json


class MultiModelLLMService:

    def __init__(self):
        self.models_config = {
            "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
            "mamay": "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0",
            "phi": "microsoft/Phi-3-mini-4k-instruct",
            "zephyr": "HuggingFaceH4/zephyr-7b-beta"
        }

        self.models = {}
        self.tokenizers = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._load_models()

    # -------------------------
    def _load_models(self):
        for name, model_path in self.models_config.items():
            print(f"Loading {name}...")

            tokenizer = AutoTokenizer.from_pretrained(model_path)

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )

            self.models[name] = model
            self.tokenizers[name] = tokenizer

    # -------------------------
    def _build_prompt(self, text: str) -> str:
        return f"""
дай відповідь СТРОГО у JSON форматі:

{{
 "ThreatType": "БПЛА або Ракета або Unknown",
 "CurrentLocation": "Назва або Unknown",
 "Direction": "Назва або Unknown",
 "Count": "Число або Unknown"
}}

Проаналізуй повідомлення про повітряну загрозу.

Повідомлення:
{text}

Правила:
1. ThreatType:
   - "БПЛА" якщо згадується шахед, дрон, БПЛА
   - "Ракета" якщо згадується ракета

2. CurrentLocation — місто або населений пункт ЗВІДКИ летить об'єкт.

3. Direction — місто або напрямок КУДИ летить.

4. Count — кількість об'єктів. Якщо не вказано → 1.

5. Якщо інформації немає → "Unknown".

6. Поверни ТІЛЬКИ JSON.
"""

    # -------------------------
    def _extract_json(self, text: str):
        if not text:
            return None

        text = text.replace("```json", "").replace("```", "")

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    # -------------------------
    def _generate(self, model, tokenizer, prompt: str):

        start = time.time()

        messages = [
            {"role": "user", "content": prompt}
        ]

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        parsed = extract_json(text)

        end = time.time()

        return {
            "result": parsed,
            "time": round(end - start, 3),
            "raw": text if parsed is None else None
        }

    # -------------------------
    def run_all(self, text: str):

        prompt = self._build_prompt(text)

        results = {}

        for name in self.models:
            try:
                results[name] = self._generate(
                    self.models[name],
                    self.tokenizers[name],
                    prompt
                )
            except Exception as e:
                results[name] = {
                    "error": str(e)
                }

        return results