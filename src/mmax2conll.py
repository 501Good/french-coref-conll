import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from pprint import pprint
from typing import Optional

import stanza
import udapi
from lxml import etree
from stanza.models.mwt.utils import resplit_mwt
from stanza.utils.conll import CoNLL
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

data_path = Path("data")
words_dir = Path("Basedata")
ana_dir = Path("Markables")
stanza_out_dir = Path("stanza_out/parcorfull")
udpipe_out_dir = Path("udpipe_out/parcorfull")
file_name = Path("000_779.mmax")

stanza_out_dir.mkdir(parents=True, exist_ok=True)
udpipe_out_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


@dataclass
class Span:
    prefix: str
    start_idx: int
    end_idx: int

    def __eq__(self, other):
        if self.start_idx == other.start_idx and self.end_idx == other.end_id:
            return True
        else:
            return False

    def __lt__(self, other):
        return True if self.end_idx < other.start_idx else False

    def __le__(self, other):
        return True if self.end_idx <= other.start_idx else False

    def __gt__(self, other):
        return True if self.start_idx > other.end_idx else False

    def __ge__(self, other):
        return True if self.start_idx >= other.end_idx else False

    def __contains__(self, item):
        if isinstance(item, int):
            if item >= self.start_idx and item <= self.end_idx:
                return True
            else:
                return False
        else:
            raise NotImplementedError

    def __len__(self):
        return self.end_idx - self.start_idx


@dataclass
class Coref:
    coref_class: str
    id: str
    span: str
    clausetype: Optional[str] = None
    vptype: Optional[str] = None
    type: Optional[str] = None
    type_of_pronoun: Optional[str] = None
    npmod: Optional[str] = None
    mmax_level: Optional[str] = None
    split: Optional[str] = None
    comparative: Optional[str] = None
    nptype: Optional[str] = None
    antetype: Optional[str] = None
    adverbtype: Optional[str] = None
    mention: Optional[str] = None
    anacata: Optional[str] = None

    def __post_init__(self):
        other = [
            "clausetype",
            "vptype",
            "type",
            "type_of_pronoun",
            "npmod",
            "split",
            "comparative",
            "nptype",
            "antetype",
            "adverbtype",
            "mention",
            "anacata",
        ]

        if "," in self.span:
            self.span = [parse_span(s) for s in self.span.split(",")]
        else:
            self.span = [parse_span(self.span)]

        self.misc = ",".join(
            f"{field.name}:{getattr(self, field.name).replace(' ', '+').replace('-', '%2D')}"
            for field in fields(self)
            if field.name in other and getattr(self, field.name) is not None
        )


@dataclass
class CorpusToken:
    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None


@dataclass
class CoNLLTokenLocation:
    sent_id: int
    token_id: int

    def __hash__(self):
        return hash((self.sent_id, self.token_id))

    def __repr__(self):
        return f"{self.sent_id}#{self.token_id}"


def parse_span(span: str) -> Span:
    if ".." in span:
        sent_span_start, sent_span_end = span.split("..")
    else:
        sent_span_start = sent_span_end = span

    prefix = sent_span_start.split("_")[0]
    sent_span_start_idx = int(sent_span_start.split("_")[-1])
    sent_span_end_idx = int(sent_span_end.split("_")[-1])

    return Span(prefix, sent_span_start_idx, sent_span_end_idx)


def parse_words(file_path: str | Path) -> dict[str, CorpusToken]:
    words_dict: dict[str, str] = {}
    words = etree.parse(file_path)
    for word in words.findall(".//word"):
        words_dict[word.get("id")] = CorpusToken(text=word.text)
    return words_dict


def parse_sentences(file_path: str | Path) -> dict[int, Span]:
    sents_dict: dict[int, Span] = {}
    sents = etree.parse(file_path)
    for sent in sents.findall(".//{*}markable"):
        order_id = sent.get("orderid")
        sents_dict[int(order_id)] = parse_span(sent.get("span"))
    sents_dict = dict(sorted(sents_dict.items()))
    return sents_dict


def parse_coref(file_path: str | Path) -> dict[list[Coref]]:
    coref_dict: dict[list[Coref]] = defaultdict(list)
    corefs = etree.parse(file_path)
    for coref in corefs.findall(".//{*}markable"):
        coref_dict[coref.attrib["coref_class"]].append(Coref(**coref.attrib))
    return dict(sorted(coref_dict.items()))


def construct_entity_string(misc_text, ent_id, open=True):
    ent_str = f"e{ent_id}"
    if open:
        ent_str = "(" + ent_str
    else:
        ent_str = ent_str + ")"
    if "Entity=" not in misc_text:
        ent_str = "Entity=" + ent_str
    return ent_str


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("corpus_name", type=str, choices=("DiscoMT", "TED"), help="The name of the corpus to process.")

    args = argparser.parse_args()

    corpus_name = args.corpus_name
    corpus_path = Path(f"parcor-full/corpus/{corpus_name}/FR")

    nlp_tokenized = stanza.Pipeline(lang="fr", processors="tokenize,mwt")
    nlp = stanza.Pipeline(lang="fr", processors="tokenize,pos,lemma,depparse", tokenize_pretokenized=True)

    for file_path in (pbar := tqdm(Path(data_path / corpus_path).glob("*.mmax"))):
        with logging_redirect_tqdm():
            pbar.set_description(f"Processing {file_path}")
            mmax = etree.parse(file_path)
            words_file_name = mmax.find(".//words").text

            words_dict = parse_words(data_path / corpus_path / words_dir / words_file_name)
            sents_dict = parse_sentences(data_path / corpus_path / ana_dir / f"{file_path.stem}_sentence_level.xml")
            coref_dict = parse_coref(data_path / corpus_path / ana_dir / f"{file_path.stem}_coref_level.xml")

            mmax2conll_ids = {}

            document: list[list[str]] = []
            for k, v in sents_dict.items():
                tokens = []
                for i, idx in enumerate(range(v.start_idx, v.end_idx + 1), start=1):
                    tokens.append(words_dict[f"{v.prefix}_{idx}"].text)
                document.append(tokens)

            doc = resplit_mwt(document, nlp_tokenized)

            doc = nlp(doc)

            total_word_count = 1
            for sent_id, sent in enumerate(doc.sentences):
                sent.sent_id = f"{corpus_name}-{file_path.stem}-{sent_id}"
                sent.doc_id = f"{corpus_name}-{file_path.stem}"
                for token_id, token in enumerate(sent.tokens):
                    # We will need the mapping from the original word ids to the sentence and token ids in the CoNLL output later.
                    # `stanza` stores all the token ids in a tuple, so a single-word token will have an id like `(1,)`.
                    # and a multi-word token will have an id like `(1, 2)`.
                    # In French, the multi-word tokens are amalgams (ex. aux = à + les).
                    # In case the mention starts with a multi-word token in the original file, we don't want to include a preposition in it.
                    # Thus the last token id is taken every time.
                    mmax2conll_ids[f"word_{total_word_count}"] = CoNLLTokenLocation(
                        sent_id=sent_id, token_id=token.id[-1]
                    )
                    total_word_count += 1

            # for i, token in enumerate(doc.iter_tokens(), start=1):
            #     words_dict[f"word_{i}"] = CorpusToken(token.text, start_char=token.start_char, end_char=token.end_char)

            # udapi only works with files, so we have to write out the result from stanza into a file first
            CoNLL.write_doc2conll(doc, f"{stanza_out_dir}/{corpus_name}-{file_path.stem}.conllu")

            ud_doc = udapi.Document(f"{stanza_out_dir}/{corpus_name}-{file_path.stem}.conllu")

            trees = list(ud_doc.trees)
            words = [tree.descendants for tree in trees]
            ents = []
            for cluster, mentions in coref_dict.items():
                # All the singeltons in ParCorFull have the coref_class "empty"
                if cluster == "empty":
                    for mention in mentions:
                        e = ud_doc.create_coref_entity()
                        if len(mention.span) == 1:
                            word_start = mmax2conll_ids[f"word_{mention.span[0].start_idx}"]
                            word_end = mmax2conll_ids[f"word_{mention.span[0].end_idx}"]
                            # Not sure if this is the best way to manage mentions but it works
                            m = e.create_mention(
                                words=words[word_start.sent_id][word_start.token_id - 1 : word_end.token_id]
                            )
                            m.other = mention.misc
                        else:
                            mention_words = []
                            for span in mention.span:
                                word_start = mmax2conll_ids[f"word_{span.start_idx}"]
                                word_end = mmax2conll_ids[f"word_{span.end_idx}"]
                                mention_words.extend(
                                    words[word_start.sent_id][word_start.token_id - 1 : word_end.token_id]
                                )
                            m = e.create_mention(words=mention_words)
                            m.other = mention.misc
                        ents.append(e)
                else:
                    e = ud_doc.create_coref_entity()
                    for mention in mentions:
                        if len(mention.span) == 1:
                            word_start = mmax2conll_ids[f"word_{mention.span[0].start_idx}"]
                            word_end = mmax2conll_ids[f"word_{mention.span[0].end_idx}"]
                            try:
                                m = e.create_mention(
                                    words=words[word_start.sent_id][word_start.token_id - 1 : word_end.token_id]
                                )
                                m.other = mention.misc
                            # Ignore mentions spanning across several sentences
                            except IndexError as err:
                                logging.error(
                                    f"Mention {mention.id} spans across several sentence, which is not supported!"
                                )
                                # tqdm.write(str(e))
                                # raise e
                        else:
                            prev_sent_id = mmax2conll_ids[f"word_{mention.span[0].start_idx}"].sent_id
                            mention_words = []
                            for span in mention.span:
                                word_start = mmax2conll_ids[f"word_{span.start_idx}"]
                                word_end = mmax2conll_ids[f"word_{span.end_idx}"]
                                # Since CoNLL-U does not support mentions that span across several sentences,
                                # we split the into separate mentions belonging to the same entity
                                if word_start.sent_id != prev_sent_id:
                                    m = e.create_mention(words=mention_words)
                                    m.other = mention.misc
                                    mention_words = []
                                mention_words.extend(
                                    words[word_start.sent_id][word_start.token_id - 1 : word_end.token_id]
                                )
                                prev_sent_id = word_start.sent_id
                            m = e.create_mention(words=mention_words)
                            m.other = mention.misc
                    ents.append(e)

            # Add the document name to the first sentence of the document
            ud_doc[0].trees[0].newdoc = f"{corpus_name}-{file_path.stem}"

            # Somewhere along the processing pipeline (I guess in stanza), the FEATS are not properly sorted.
            # This causes the UD validation to fail.
            # I cannot find where exactly the bad sorting happens so the easiest solution for now it just to resort the FEATS post hoc.
            for node in ud_doc.nodes:
                node_feats = str(node.feats)
                sorted_feats = "|".join(sorted(node_feats.split("|"), key=lambda x: x.lower()))
                if node_feats != sorted_feats:
                    logging.info(
                        f"Re-sorted feats are different from the original ones: {node_feats} -> {sorted_feats}"
                    )
                    node.feats = sorted_feats

            ud_doc.store_conllu(f"{udpipe_out_dir}/{corpus_name}-{file_path.stem}.conllu")


if __name__ == "__main__":
    main()
