import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch
from datasets import Dataset
from tqdm import tqdm

print("Reading the czech_train dataframe")
df_train = pd.read_json("czech_train.json")
model_name = "google/mt5-base"
max_length = 382  #default 384 modified for T5 preprocessing 382
doc_stride = 128

tokenizer = T5TokenizerFast.from_pretrained(model_name)

print("Checking if tokenizer knows all answers")


def check_answers():
    unk_token = tokenizer.special_tokens_map.get("unk_token")
    data = []
    count = 0

    for row in tqdm(df_train.iloc, total=df_train.shape[0]):
        tokenized_row = tokenizer(row["answers"]["text"],
                                  max_length=max_length,
                                  truncation="only_second",
                                  stride=doc_stride,
                                  return_overflowing_tokens=True,
                                  return_offsets_mapping=True)

        for i in tokenized_row["input_ids"]:
            if unk_token in i:
                count += 1
                continue
        data.append([row.answers["text"], row.context, row.question, row.id, row.title])
    print("\nDeleted {} questions".format(count))
    data_frame = pd.DataFrame(data, columns=["answer_text", "question", "context", "id", "title"])
    return data_frame


data_df = check_answers()


def tokenize_row(row):
    #Tokenized "context:"
    tokenized_context_text = [19730, 267]
    
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
    tokenized_row["labels"] = []
    num_tokens_until_context = tokenized_row["input_ids"][0].index(1) + 1

    for i in range(len(tokenized_row["input_ids"])):
        tokenized_row["input_ids"][i].insert(num_tokens_until_context, tokenized_context_text[1])
        tokenized_row["input_ids"][i].insert(num_tokens_until_context, tokenized_context_text[0])
        tokenized_row["attention_mask"][i].insert(num_tokens_until_context, 1)
        tokenized_row["attention_mask"][i].insert(num_tokens_until_context, 1)
        tokenized_row["offset_mapping"][i].insert(num_tokens_until_context, (0, 0))
        tokenized_row["offset_mapping"][i].insert(num_tokens_until_context, (0, 0))
        
        pos = 1
        first_char_index = tokenized_row["offset_mapping"][i][num_tokens_until_context + 2:][0][0]
        last_char_index = tokenized_row["offset_mapping"][i][num_tokens_until_context + 2:][-pos]
        while last_char_index[1] == 0:
            pos += 1
            last_char_index = tokenized_row["offset_mapping"][i][num_tokens_until_context + 2:][-pos]
        last_char_index = last_char_index[1]

        if first_char_index <= answer_start <= last_char_index:
            tokenized_row["labels"].append(tokenizer(row["answers"]["text"][0])["input_ids"])
        else:
            tokenized_row["labels"].append([2])

    return tokenized_row


print("Converting dataframe to dataset")
train_dataset = Dataset.from_pandas(df_train)
train_dataset = train_dataset.remove_columns("__index_level_0__")
train_dataset = train_dataset.train_test_split(test_size=0.1)

print("Tokenizing dataset")
tokenized_train_dataset = train_dataset.map(tokenize_row, remove_columns=train_dataset["train"].column_names)

print("Deleting train dataset and train dataframe")
del train_dataset
del df_train


def unnest_dataset(dataset):
  pandas_data = []
  for k in dataset.keys():
    for inp in tqdm(dataset[k], total=dataset[k].num_rows):
      for ind in range(len(inp["input_ids"])):
        #print(tokenizer.decode(inp["input_ids"][ind]))
        pandas_data.append([inp["input_ids"][ind],inp["attention_mask"][ind],inp["labels"][ind]])
  df = pd.DataFrame(pandas_data, columns=["input_ids","attention_mask","labels"])
  return df


print("Creating final training dataset")
final_df = unnest_dataset(tokenized_train_dataset)

del tokenized_train_dataset

tokenized_train_dataset = Dataset.from_pandas(final_df)
tokenized_train_dataset = tokenized_train_dataset.train_test_split(test_size=0.1)

del tokenized_train_dataset

print("Loading model")
model = T5ForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda:0")
model.to(device)
print("Training on:", torch.cuda.get_device_name(device))

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model
    )

train_args = Seq2SeqTrainingArguments(
              output_dir = "T5_multilingual_finetuned_sqad",
              evaluation_strategy = "epoch",
              learning_rate = 3e-5,
              num_train_epochs = 3,
              weight_decay = 0.1,
              per_device_train_batch_size = 16,
              per_device_eval_batch_size = 16,
              save_total_limit=2
              )

trainer = Seq2SeqTrainer(
          model=model,
          args=train_args,
          train_dataset=tokenized_train_dataset["train"],
          eval_dataset=tokenized_train_dataset["test"],
          tokenizer=tokenizer,
          data_collator=data_collator,
          )

trainer.train()

model.save_pretrained("T5_multilingual_finetuned_sqad/T5-multilingual-finetuned-sqad")
exit()
