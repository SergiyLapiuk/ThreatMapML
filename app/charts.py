import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def main():
    sns.set(style="whitegrid")

    data_summary = pd.DataFrame({
        "Model": ["qwen", "mistral", "mamay", "phi", "zephyr"],
        "Accuracy": [0.588, 0.725, 0.825, 0.625, 0.85],
        "Time": [20.834, 349.789, 196.847, 175.189, 698.054]
    })

    data_cases = pd.DataFrame({
        "TestCase": list(range(1, 21)) * 5,
        "Model": (
            ["qwen"] * 20 +
            ["mistral"] * 20 +
            ["mamay"] * 20 +
            ["phi"] * 20 +
            ["zephyr"] * 20
        ),
        "Accuracy": (
            [0.75,0.75,0.75,0.25,0.75,0.75,0.75,0.75,0.5,0.5,
             0.5,0.5,0.5,0.75,0.25,0.5,0.75,0.5,0.25,0.75] +

            [1,0.75,1,0.5,1,0.5,1,0.5,0.75,0.5,
             0.75,0.5,1,0.75,0.25,0.5,1,1,0.5,0.75] +

            [1,1,1,0.5,1,1,0.75,1,0.5,0.5,
             1,0.5,1,1,0.5,0.5,1,1,1,0.75] +

            [1,0.75,0.75,0.5,0,1,0.5,0.5,0.75,0.5,
             0.5,0.5,0.75,1,0.5,0.5,0.5,0.75,0.25,1] +

            [1,1,1,0.5,1,1,1,0.5,0.5,0.5,
             1,0.5,1,1,1,0.5,1,1,1,1]
        )
    })

    plt.figure(figsize=(8,5))
    sns.barplot(data=data_summary, x="Model", y="Accuracy")
    plt.title("Average Accuracy per Model")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.barplot(data=data_summary, x="Model", y="Time")
    plt.title("Average Response Time")
    plt.ylabel("Seconds")
    plt.show()

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data_summary, x="Time", y="Accuracy", hue="Model", s=100)

    for i in range(len(data_summary)):
        plt.text(
            data_summary["Time"][i],
            data_summary["Accuracy"][i],
            data_summary["Model"][i]
        )

    plt.title("Speed vs Accuracy")
    plt.show()

    plt.figure(figsize=(12,6))
    sns.lineplot(data=data_cases, x="TestCase", y="Accuracy", hue="Model", marker="o")
    plt.title("Accuracy per Test Case")
    plt.show()

    pivot = data_cases.pivot(index="TestCase", columns="Model", values="Accuracy")

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f")
    plt.title("Теплова карта точності")
    plt.xlabel("Модель")
    plt.ylabel("Тестовий випадок")
    plt.show()

    data_summary["Efficiency"] = data_summary["Accuracy"] / data_summary["Time"]

    plt.figure(figsize=(8,5))
    sns.barplot(data=data_summary, x="Model", y="Efficiency")
    plt.title("Efficiency (Accuracy / Time)")
    plt.show()

if __name__ == "__main__":
    main()