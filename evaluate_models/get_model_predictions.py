from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer
import torch
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings('ignore')

"""
Script for evaluating larger model like XML RoBERTa, creates top 20 predictions for each question in test dataset
Process is similar as in get_model_predictions.ipynb
"""

model_name = "xlm-roberta-large"
max_length = 384
doc_stride = 128
n_best_answers = 20

print("Loading tokenizer and model:")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained("roberta-multilingual-finetuned-sqad", local_files_only=True)
device = torch.device("cuda:0")
print(torch.cuda.get_device_name(device))
model.to(device)

print("Reading test dataset:")
test_df = pd.read_json("/nlp/projekty/qa_2022/training_models/czech_test.json")

def tokenize_row(row):
    tokenized_row = tokenizer(
        row["question"],
        row["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = tokenized_row.pop("overflow_to_sample_mapping")
    tokenized_row["example_id"] = []
    for i in range(len(tokenized_row["input_ids"])):
        sample_idx = sample_map[i]
        tokenized_row["example_id"].append(row["id"][sample_idx])
        offset = tokenized_row["offset_mapping"][i]
        #print(len(tokenized_row["offset_mapping"][i]))
        for k, v in enumerate(tokenized_row["offset_mapping"][i]):
            #print(tokenized_row.sequence_ids(i)[k], end=" ")
            if tokenized_row.sequence_ids(i)[k] == 1:
                tokenized_row["offset_mapping"][i][k] = v
            else:
                tokenized_row["offset_mapping"][i][k] = None
    return tokenized_row

print("Tokenizing dataset:")
dataset_test = Dataset.from_pandas(test_df)
dataset_test = dataset_test.remove_columns(["__index_level_0__"])
tokenized_test =  dataset_test.map(tokenize_row, batched=True, remove_columns=dataset_test.column_names)

from transformers import default_data_collator
data_collator = default_data_collator

trainer = Trainer(
    model,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

preds = trainer.predict(tokenized_test)

from collections import OrderedDict, defaultdict
from tqdm import tqdm


def preprocess_predictions(dataset, tokenized_dataset, predictions):
    start_logits, end_logits = predictions
    predictions = OrderedDict()
    example_id_to_index = {k: i for i, k in enumerate(dataset["id"])}
    # Create dict for mapping each tokenized input to question index
    # so we know which tokenized input coresponds to which question
    features_per_example = defaultdict(list)
    for i, feature in enumerate(tqdm(tokenized_dataset)):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    # Loop through all of the questions
    for example_index, example in enumerate(tqdm(dataset)):
        valid_answers = []
        # Loop through all of the inputs in current question
        for input_index in features_per_example[example_index]:
            current_start_logits = start_logits[input_index]
            current_end_logits = end_logits[input_index]
            offset_mapping = tokenized_dataset[input_index]["offset_mapping"]
            start_indexes = np.argsort(current_start_logits)[-1: -n_best_answers - 1: -1].tolist()
            end_indexes = np.argsort(current_end_logits)[-1: -n_best_answers - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offset_mapping) \
                            or end_index >= len(offset_mapping) \
                            or offset_mapping[start_index] is None \
                            or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    if start_index <= end_index:
                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        valid_answers.append(
                            {"score": current_start_logits[start_index] + current_end_logits[end_index],
                             "text": example["context"][start_char: end_char]})

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        predictions[example["id"]] = best_answer["text"]

    # print(features_per_example)

    return predictions

print("Getting predictions:")
prepd_predictions = preprocess_predictions(dataset_test, tokenized_test, preds.predictions)

prepd_predictions = dict(prepd_preds)

print("Saving predictions")
with open('RoBERTa_multilingual_predictions.json', 'wb') as fp:
    pickle.dump(prepd_predictions, fp)
