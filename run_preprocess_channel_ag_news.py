import json

import fire
from datasets import load_dataset


def main(sample=True):
    dataset = load_dataset("ag_news")
    label_words = ["World", "Sports", "Business", "Science and technology"]

    # Sample demonstrations
    demonstrations = dataset["train"].train_test_split(
        train_size=4, seed=42, stratify_by_column="label"
    )["train"]

    # Create demonstration prompt
    prompt = ""
    for demo in demonstrations:
        input_prompt = f"Article: {demo['text']}\n"
        label_prompt = f"Category: {label_words[demo['label']]}\n"
        prompt += f"{label_prompt}{input_prompt}"
    prompt = prompt + "Category: {label_word}\nArticle:"

    # Create prompt and completion
    data = []
    test_dataset = dataset["test"].select([0, 1, 2, 3]) if sample else dataset["test"]
    for test in test_dataset:
        data.append(
            {
                "prompt": prompt,
                "completion": test["text"],
                "label_words": label_words,
                "ground_truth": label_words[test["label"]],
            }
        )

    # Save data
    data = sorted(data, key=lambda x: -len(x["prompt"] + x["completion"]))
    with open("samples/inputs_channel_ag_news.json", "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
