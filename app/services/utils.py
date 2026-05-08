import json
import re


def extract_json(text: str):
    if not text:
        return None

    # прибираємо markdown ```json
    text = text.replace("```json", "").replace("```", "")

    # 🔥 знаходимо всі JSON-об'єкти
    matches = re.findall(r"\{[\s\S]*?\}", text)

    if not matches:
        return None

    # 👉 беремо ДРУГИЙ якщо є, інакше останній
    json_str = matches[1] if len(matches) > 1 else matches[-1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            json_str = json_str.replace("'", '"')
            return json.loads(json_str)
        except:
            return None