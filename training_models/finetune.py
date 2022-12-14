import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments, default_data_collator
import torch
from datasets import Dataset
from tqdm import tqdm

"""
Load SQAD dataset, tokenizer and set name of the finetuned model and max_length and doc_stride for tokenization
"""

model_name = "xlm-roberta-large"
df_train = pd.read_json("czech_train.json")
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = 384
doc_stride = 128
roberta = True if model_name == "xlm-roberta-large" else False


"""
Since RoBERTa doesn't have token_type_ids so this function creates them
"""

def create_token_type_ids(ex):
    token_type_ids = []
    for i in range(len(ex.input_ids)):
        si = np.array(ex.sequence_ids(i))
        si = np.where(si == None, 0, si).tolist()
        token_type_ids.append(si)
    return token_type_ids


def context_start_token_index(ex):
    return [x.index(1) for x in ex.token_type_ids]


def context_end_token_index(ex):
    return [(len(x) - x[::-1].index(1)) for x in ex.token_type_ids]

"""
Tokenize row, split into chunks and set starting and ending tokens of question
"""

def get_tokenized_row(row, hf_dataset=True):
    tokenized_row = tokenizer(row["question"],
                              row["context"],
                              max_length=max_length,
                              truncation="only_second",
                              padding="max_length",
                              stride=doc_stride,
                              return_overflowing_tokens=True,
                              return_offsets_mapping=True)
    tokenized_row["start_positions"] = []
    tokenized_row["end_positions"] = []
    if roberta:
        tokenized_row["token_type_ids"] = create_token_type_ids(tokenized_row)
    context_starts = context_start_token_index(tokenized_row)
    context_ends = context_end_token_index(tokenized_row)
    index = tokenized_row["overflow_to_sample_mapping"]


    for i, o in enumerate(tokenized_row["offset_mapping"]):

        if hf_dataset:
            start_ch = row["answers"][index[i]]["answer_start"][0]
            end_ch = start_ch + len(row["answers"][index[i]]["text"][0])
        else:
            start_ch = row["answers"]["answer_start"][0]
            end_ch = start_ch + len(row["answers"]["text"][0])

        cls_ind = tokenized_row["input_ids"][i].index(tokenizer.cls_token_id)
        start = context_starts[i]
        end = context_ends[i] - 2
        if start_ch < o[start][0] or end_ch > o[end][1]:
            tokenized_row["start_positions"].append(cls_ind)
            tokenized_row["end_positions"].append(cls_ind)
        else:
            while o[start][0] < start_ch:
                start += 1
            while o[end][1] > end_ch:
                end -= 1
                if start > end:
                    end += 1
                    break
            tokenized_row["start_positions"].append(start)
            tokenized_row["end_positions"].append(end)
    return tokenized_row


"""
Delete rows with answers not recognizable by model's tokenizer
"""

count = 0
data = []
for row in tqdm(df_train.iloc, total=df_train.shape[0], desc="Checking all answers and answers in contexts"):
    test = get_tokenized_row(row, hf_dataset=False)
    answers = []
    for i, z in enumerate(zip(test["start_positions"],test["end_positions"])):
        x = test["input_ids"][i][z[0]:z[1]+1]
        answers.append(tokenizer.decode(x))
    if row.answers["text"][0].replace(' ', '').lower() not in [x.replace(' ','').lower() for x in answers]:
        count += 1
        continue
    data.append([row.answers, row.question, row.context, row.id, row.title])
print("Deleted",count,"questions")
data_df = pd.DataFrame(data, columns=["answers", "question", "context", "id", "title"])

"""
Load SQAD into huggingface dataset and split into train and validation
"""
train_dataset = Dataset.from_pandas(data_df)
train_dataset = train_dataset.train_test_split(test_size=0.1)
tokenized_train_dataset = train_dataset.map(get_tokenized_row, remove_columns=train_dataset["train"].column_names, batched=True)

"""
Remove unused columns
"""
tokenized_train_dataset["train"] = tokenized_train_dataset["train"].remove_columns("offset_mapping")
tokenized_train_dataset["test"] = tokenized_train_dataset["test"].remove_columns("offset_mapping")
tokenized_train_dataset["train"] = tokenized_train_dataset["train"].remove_columns("overflow_to_sample_mapping")
tokenized_train_dataset["test"] = tokenized_train_dataset["test"].remove_columns("overflow_to_sample_mapping")
tokenized_train_dataset["train"] = tokenized_train_dataset["train"].remove_columns("token_type_ids")
tokenized_train_dataset["test"] = tokenized_train_dataset["test"].remove_columns("token_type_ids")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
Load model and set training arguments
"""

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model.to(device)

training_arg = TrainingArguments(
                output_dir = "RoBERTa_multilingual_finetuned_sqad",
                evaluation_strategy = "epoch",
                learning_rate = 3e-5,
                num_train_epochs = 3,
                weight_decay = 0.1,
                per_device_train_batch_size = 16,
                per_device_eval_batch_size = 16,
                save_total_limit=2
                )

data_collator = default_data_collator

trainer = Trainer(
            model,
            training_arg,
            data_collator=data_collator,
            train_dataset=tokenized_train_dataset["train"],
            eval_dataset=tokenized_train_dataset["test"],
            tokenizer=tokenizer)

trainer.train()

model.save_pretrained("RoBERTa_multilingual_finetuned_sqad/roberta-multilingual-finetuned-sqad")
exit()
