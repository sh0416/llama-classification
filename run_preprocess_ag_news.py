import json

import fire
from datasets import load_dataset


def main(sample=True):
    dataset = load_dataset("ag_news")
    label_words = ["World", "Sports", "Business", "Science and technology"]

    demonstrations = dataset["train"].train_test_split(
        train_size=4, seed=42, stratify_by_column="label"
    )["train"]

    prompts_and_completions = []
    test_dataset = dataset["test"].select([0, 1, 2, 3]) if sample else dataset["test"]
    for test in test_dataset:
        prompt = ""
        for demo in demonstrations:
            input_prompt = f"""Article: {demo["text"]}\n"""
            label_prompt = f"Category: {label_words[demo['label']]}\n"
            prompt += f"""{input_prompt}{label_prompt}"""
        prompt += f"Article: {test['text']}\nCategory:"
        prompts_and_completions.append(
            {
                "prompt": prompt,
                "candidate_completions": label_words,
                "completion": label_words[test["label"]],
            }
        )

    prompts_and_completions = sorted(
        prompts_and_completions, key=lambda x: -len(x["prompt"] + x["completion"])
    )
    with open("samples/text_completions_ag_news.json", "w") as f:
        json.dump(prompts_and_completions, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
