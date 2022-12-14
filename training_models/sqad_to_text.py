import pandas as pd
import re

def get_docu(path):
    sentences = []
    sentence = []
    g = False
    with open(path, 'r') as f:
        for line in f:
            #start of a sentence
            if line[:2] == "<s":
                if "desamb" in line:
                    continue
                sentence = []
            elif line[:3] == "</s":
                sentences.append(" ".join(sentence))
                sentence = []
            elif line[:2] == "<g":
                if not sentence:
                    continue
                g = True
                continue
            else:
                w = re.match("^[^\\s]+", line)
                if not w:
                   continue
                if g:
                    sentence[-1] += w.group(0)
                else:
                    sentence.append(w.group(0))
            g = False
        if sentence:
            sentences.append(' '.join(sentence))

    return '\n'.join(sentences)

def main():
    data_path = "/nlp/projekty/sqad/sqad_v3/data/"
    print()
    df = pd.DataFrame(columns=["question", "text", "answer", "answer_sentence"])
    for i in range(1, 13474):
        if i % 1000 == 0:
            print(f"Processing {data_path}{i:06d}")
        question = get_docu(f"{data_path}{i:06d}/01question.vert")
        text = get_docu(f"{data_path}{i:06d}/03text.vert")
        question_selection = get_docu(f"{data_path}{i:06d}/06answer.selection.vert")
        exact_answer = get_docu(f"{data_path}{i:06d}/09answer_extraction.vert")
        df.loc[i] = question, text, exact_answer, question_selection
    print("Saving dataframe to csv")
    df.to_csv("sqad_dataframe.csv")
    print("Done :)")

if __name__ == "__main__":
    main()
