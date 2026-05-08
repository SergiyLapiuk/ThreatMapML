from app.services.multi_model_service import MultiModelLLMService

def evaluate(pred, expected):
    if pred is None:
        return 0

    score = 0
    total = 4

    if pred.get("ThreatType") == expected["ThreatType"]:
        score += 1

    if pred.get("CurrentLocation") == expected["CurrentLocation"]:
        score += 1

    if pred.get("Direction") == expected["Direction"]:
        score += 1

    if str(pred.get("Count")) == str(expected["Count"]):
        score += 1

    return score / total

def main():
    service = MultiModelLLMService()

    test_cases = [
        {
            "text": "2 ракети з Херсона на Миколаїв",
            "expected": {
                "ThreatType": "Ракета",
                "CurrentLocation": "Херсон",
                "Direction": "Миколаїв",
                "Count": 2
            }
        },
        {
            "text": "БПЛА рухається з Миколаєва в Одесу",
            "expected": {
                "ThreatType": "БПЛА",
                "CurrentLocation": "Миколаїв",
                "Direction": "Одеса",
                "Count": 1
            }
        },
        {
            "text": "Ракета з Криму летить на Київ",
            "expected": {
                "ThreatType": "Ракета",
                "CurrentLocation": "Крим",
                "Direction": "Київ",
                "Count": 1
            }
        },
        {
            "text": "Ракета летить на Миколаїв",
            "expected": {
                "ThreatType": "Ракета",
                "CurrentLocation": "Unknown",
                "Direction": "Миколаїв",
                "Count": 1
            }
        },
        {
            "text": "2 ракети з Херсона на Миколаїв та ще 1 БПЛА з Криму на Одесу",
            "expected": {
                "ThreatType": "Ракета",  # спірний кейс (можеш тестити multi-output пізніше)
                "CurrentLocation": "Херсон",
                "Direction": "Миколаїв",
                "Count": 2
            }
        },
        {
            "text": "3 шахеди з Запоріжжя на Дніпро",
            "expected": {
                "ThreatType": "БПЛА",
                "CurrentLocation": "Запоріжжя",
                "Direction": "Дніпро",
                "Count": 3
            }
        },
        {
            "text": "Дрон над Києвом",
            "expected": {
                "ThreatType": "БПЛА",
                "CurrentLocation": "Київ",
                "Direction": "Unknown",
                "Count": 1
            }
        },
        {
            "text": "5 ракет з Сум в Харків",
            "expected": {
                "ThreatType": "Ракета",
                "CurrentLocation": "Суми",
                "Direction": "Харків",
                "Count": 5
            }
        },
        {
            "text": "Ракети рухаються на захід",
            "expected": {
                "ThreatType": "Ракета",
                "CurrentLocation": "Unknown",
                "Direction": "захід",
                "Count": 1
            }
        },
        {
            "text": "1 БПЛА з невідомого напрямку на Одесу",
            "expected": {
                "ThreatType": "БПЛА",
                "CurrentLocation": "Unknown",
                "Direction": "Одеса",
                "Count": 1
            }
        },
        {
            "text": "Шахед з Криму рухається на Миколаїв",
            "expected": {
                "ThreatType": "БПЛА",
                "CurrentLocation": "Крим",
                "Direction": "Миколаїв",
                "Count": 1
            }
        },
        {
            "text": "Ракета з моря на Одесу",
            "expected": {
                "ThreatType": "Ракета",
                "CurrentLocation": "море",
                "Direction": "Одеса",
                "Count": 1
            }
        },
        {
            "text": "4 БПЛА з Херсона на Запоріжжя",
            "expected": {
                "ThreatType": "БПЛА",
                "CurrentLocation": "Херсон",
                "Direction": "Запоріжжя",
                "Count": 4
            }
        },
        {
            "text": "Нова загроза невідомого типу",
            "expected": {
                "ThreatType": "Unknown",
                "CurrentLocation": "Unknown",
                "Direction": "Unknown",
                "Count": 1
            }
        },
        {
            "text": "Ракета на Київ",
            "expected": {
                "ThreatType": "Ракета",
                "CurrentLocation": "Unknown",
                "Direction": "Київ",
                "Count": 1
            }
        },
        {
            "text": "2 дрони з півдня на Одесу",
            "expected": {
                "ThreatType": "БПЛА",
                "CurrentLocation": "півдня",
                "Direction": "Одеса",
                "Count": 2
            }
        },
        {
            "text": "Одна ракета з Криму в бік Києва",
            "expected": {
                "ThreatType": "Ракета",
                "CurrentLocation": "Крим",
                "Direction": "Київ",
                "Count": 1
            }
        },
        {
            "text": "БПЛА кружляє над Харковом",
            "expected": {
                "ThreatType": "БПЛА",
                "CurrentLocation": "Харків",
                "Direction": "Unknown",
                "Count": 1
            }
        },
        {
            "text": "6 ракет з Бродів на Львів",
            "expected": {
                "ThreatType": "Ракета",
                "CurrentLocation": "Броди",
                "Direction": "Львів",
                "Count": 6
            }
        },
        {
            "text": "Повітряна тривога без уточнень",
            "expected": {
                "ThreatType": "Unknown",
                "CurrentLocation": "Unknown",
                "Direction": "Unknown",
                "Count": 1
            }
        }
    ]

    stats = {}

    print("\nRunning models...\n")

    for case in test_cases:

        text = case["text"]
        expected = case["expected"]

        print(f"\nTEXT: {text}\n")

        results = service.run_all(text)

        for model_name, data in results.items():

            pred = data["result"]
            time_taken = data["time"]

            acc = evaluate(pred, expected)

            if model_name not in stats:
                stats[model_name] = {
                    "time": [],
                    "accuracy": []
                }

            stats[model_name]["time"].append(time_taken)
            stats[model_name]["accuracy"].append(acc)

            print(f"\n=== {model_name.upper()} ===")
            print("Result:", pred)
            print("Time:", time_taken)
            print("Accuracy:", acc)

    print("\nFINAL STATS:\n")

    for model, data in stats.items():
        avg_time = sum(data["time"]) / len(data["time"])
        avg_acc = sum(data["accuracy"]) / len(data["accuracy"])

        print(f"{model}:")
        print(f" Avg Time: {round(avg_time, 3)} sec")
        print(f" Avg Accuracy: {round(avg_acc, 3)}\n")


if __name__ == "__main__":
    main()