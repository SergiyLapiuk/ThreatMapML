from app.models.threat import Threat
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import json
import httpx


class LLMService:

    def __init__(self):
        model_name = "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        self.geo_cache = {}

    async def get_coordinates(self, city: str):

        if city == "Unknown":
            return None, None

        if city in self.geo_cache:
            return self.geo_cache[city]

        url = "https://nominatim.openstreetmap.org/search"

        params = {
            "q": f"{city}, Ukraine",
            "format": "json",
            "limit": 1
        }

        headers = {
            "User-Agent": "ThreatMap"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers)

        data = response.json()

        if len(data) == 0:
            return None, None

        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])

        self.geo_cache[city] = (lat, lon)

        return lat, lon

    async def analyze_message(self, text: str) -> Threat:

        prompt = f"""<bos><start_of_turn>user
дай відповідь СТРОГО у JSON форматі:

{{
 "ThreatType": "БПЛА або Ракета або Unknown",
 "CurrentLocation": "Назва або Unknown",
 "Direction": "Назва або Unknown",
 "Count": число
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
<end_of_turn><start_of_turn>model
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")

        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=200,
            temperature=0.1,
            top_k=25,
            top_p=1,
            repetition_penalty=1.1
        )

        result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        prompt_marker = "model"
        model_text = result_text.split(prompt_marker)[-1].strip()

        try:
            start_idx = model_text.find("{")
            end_idx = model_text.rfind("}") + 1

            json_str = model_text[start_idx:end_idx]

            print(json_str)

            data = json.loads(json_str)

        except Exception:
            data = {
                "ThreatType": "Unknown",
                "CurrentLocation": "Unknown",
                "Direction": "Unknown",
                "Count": 1
            }

        current_location = data.get("CurrentLocation", "Unknown")
        direction = data.get("Direction", "Unknown")

        start_lat, start_lon = await self.get_coordinates(current_location)
        end_lat, end_lon = await self.get_coordinates(direction)

        return Threat(
            ThreatType=data.get("ThreatType", "Unknown"),
            CurrentLocation=current_location,
            Direction=direction,
            Count=data.get("Count", 1),

            StartLatitude=start_lat,
            StartLongitude=start_lon,

            EndLatitude=end_lat,
            EndLongitude=end_lon,

            DetectedAt=datetime.utcnow()
        )