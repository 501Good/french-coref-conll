import argparse
import logging
from collections import defaultdict
from pathlib import Path

import conllu

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("corpus_name", type=str, choices=("ANCOR", "ParCorFull"), help="The name of the corpus to treat.")

    args = argparser.parse_args()

    corpus = args.corpus_name
    file_path = Path(f"fr_{corpus.lower()}-corefud.conllu")
    output_path = Path(f"CorefUD/CorefUD_French-{corpus}")
    output_path.mkdir(parents=True, exist_ok=True)

    raw_data = file_path.read_text()
    sentences = conllu.parse(raw_data)

    docs = []
    doc = []
    for i, sentence in enumerate(sentences):
        if i > 0 and "newdoc id" in sentence.metadata:
            docs.append(doc)
            doc = []
        doc.append(sentence)
    docs.append(doc)

    logging.info(f"A total of {len(docs)} was read")

    split_docs = defaultdict(list)
    train_counter = 1
    split_ = "train"
    for doc in docs:
        if train_counter == -1:
            split_ = "dev"
        elif train_counter == 0:
            split_ = "test"
        else:
            split_ = "train"
        if train_counter != 0 and train_counter % 8 == 0:
            split_ = "train"
            train_counter = -2
        split_docs[split_].append(doc)
        logging.info(
            f"{doc[0].metadata['newdoc id']}: {len(doc)} sents, {len([t for sent in doc for t in sent])} tokens ({split_})"
        )
        train_counter += 1

    logging.info({split_: len(docs) for split_, docs in split_docs.items()})

    # print(train_docs[0][0].serialize())
    for split_, docs in split_docs.items():
        with open(output_path / f"{file_path.stem}-{split_}.conllu", "w", encoding="utf-8") as f:
            for doc in docs:
                for sentence in doc:
                    f.write(sentence.serialize())


if __name__ == "__main__":
    main()
