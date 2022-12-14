print("importing necessities:")
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch
from datasets import Dataset
import pandas as pd
import warnings
from collections import OrderedDict, defaultdict
from tqdm import tqdm
warnings.filterwarnings('ignore')
import pickle


model_name = "google/mt5-large"

max_length = 382  
doc_stride = 128

print("Loading tokenizer and model:")
tokenizer = T5TokenizerFast.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained("T5-multilingual-finetuned-sqad", local_files_only=True)
device = torch.device("cuda:0")
print(torch.cuda.get_device_name(device))
model.to(device)

print("Reading test dataset:")
test_df = pd.read_json("/nlp/projekty/qa_2022/training_models/czech_test.json")


def tokenize_row_t5(row):
    tokenized_context_text = [19730, 267]
    #print(len(row["question"]))
    question = "question:" + row["question"]
    answer_start = row["answers"]["answer_start"][0]
    tokenized_row = tokenizer(question,
                              row["context"],
                              max_length=max_length,
                              truncation="only_second",
                              padding="max_length",
                              stride=doc_stride,
                              return_overflowing_tokens=True,
                              return_offsets_mapping=True)
    num_tokens_until_context = tokenized_row["input_ids"][0].index(1) + 1
    for i in range(len(tokenized_row["input_ids"])):
        tokenized_row["input_ids"][i].insert(num_tokens_until_context, tokenized_context_text[1])
        tokenized_row["input_ids"][i].insert(num_tokens_until_context, tokenized_context_text[0])
        tokenized_row["attention_mask"][i].insert(num_tokens_until_context, 1)
        tokenized_row["attention_mask"][i].insert(num_tokens_until_context, 1)
        tokenized_row["offset_mapping"][i].insert(num_tokens_until_context, (0, 0))
        tokenized_row["offset_mapping"][i].insert(num_tokens_until_context, (0, 0))
    tokenized_row["example_id"] = [row["id"] for _ in range(len(tokenized_row["input_ids"]))]
    return tokenized_row

print("Tokenizing dataset:")
dataset_test = Dataset.from_pandas(test_df)
dataset_test = dataset_test.remove_columns(["__index_level_0__"])
tokenized_test =  dataset_test.map(tokenize_row_t5, remove_columns=dataset_test.column_names)

del test_df


def unnest_dataset(dataset):
    pandas_data = []
    for inp in tqdm(dataset, total=dataset.num_rows):
        for ind in range(len(inp["input_ids"])):
            pandas_data.append([inp["input_ids"][ind],inp["attention_mask"][ind], inp["example_id"][ind]])
    df = pd.DataFrame(pandas_data, columns=["input_ids","attention_mask", "example_id"])
    return df

print("Unnesting dataset:")
tokenized_test_final = unnest_dataset(tokenized_test)
tokenized_test_final = Dataset.from_pandas(tokenized_test_final)


def get_predictions_t5(generated_beam_outputs):
    #print(tokenizer.decode(2))
    predictions = []
    for i, beam_output in enumerate(generated_beam_outputs.sequences):
        #print(beam_output)
        unk_token_check = beam_output[1] == 2
        if unk_token_check:
            continue
        #print(beam_outputs.sequences_scores)
        predictions.append({"text": tokenizer.decode(beam_output,skip_special_tokens=True),
                            "score": generated_beam_outputs.sequences_scores[i].item()})
    return predictions


def preprocess_preds_t5(dataset, tokenized_dataset):
    example_id_to_index = {k: i for i, k in enumerate(dataset["id"])}
    predictions = OrderedDict()
    features_per_example = defaultdict(list)
    for i, feature in enumerate(tqdm(tokenized_dataset)):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    for example_index, example in enumerate(tqdm(dataset)):
        answers = []
        for input_index in features_per_example[example_index]:
            beam_outputs = model.generate(torch.tensor([tokenized_dataset[input_index]["input_ids"]]).cuda(),
                                         num_beams=20,
                                         num_return_sequences=20,
                                         output_scores=True,
                                         return_dict_in_generate=True,
                                         early_stopping=True)
            answers += get_predictions_t5(beam_outputs)
        best_answers = sorted(answers, key=lambda x: x["score"], reverse=True)[:20]
        predictions[example["id"]] = best_answers
    return predictions

print("Getting predictions:")
prepd_preds = preprocess_preds_t5(dataset_test, tokenized_test_final)

prepd_predictions = dict(prepd_preds)
print("Saving predictions")
with open('T5_multilingual_predictions.json', 'wb') as fp:
    pickle.dump(prepd_predictions, fp)


exit()
