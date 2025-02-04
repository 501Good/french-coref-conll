import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Any, Optional

import stanza
import udapi
from lxml import etree
from stanza.models.mwt.utils import resplit_mwt
from stanza.utils.conll import CoNLL
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


@dataclass
class CoNLLTokenLocation:
    sent_id: int
    token_id: int

    def __hash__(self):
        return hash((self.sent_id, self.token_id))

    def __repr__(self):
        return f"{self.sent_id}#{self.token_id}"


class WordIndex:
    def __init__(self, section: int = 0, utterance: int = 0, word: int = 0):
        self._section = section
        self._utterance = utterance
        self._word = word

    @classmethod
    def from_string(cls, string: str) -> None:
        """Initialise a word index from a string like "#s19.u20.w9"."""
        try:
            if string.endswith(".dash"):
                s, u, w, _ = string.split(".")
            else:
                s, u, w = string.split(".")
        except ValueError:
            raise ValueError(f'The string must be in format "#s0.u0.w0" for got "{string}" instead!')
        if s.startswith("#"):
            s = s[1:]
        return cls(int(s[1:]), int(u[1:]), int(w[1:]))
    
    @property
    def s(self):
        return self._section
    
    @s.setter
    def s(self, value):
        self._section = int(value)

    @property
    def u(self):
        return self._utterance
    
    @u.setter
    def u(self, value):
        self._utterance = int(value)

    @property
    def w(self):
        return self._word
    
    @w.setter
    def w(self, value):
        self._word = int(value)

    def __repr__(self):
        return f"s{self.s}.u{self.u}.w{self.w}"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, WordIndex):
            return self.s == other.s and self.u == other.u and self.w == other.w
        return False


data_path = Path("data")

stanza_out_dir = Path("stanza_out/ancor")
udpipe_out_dir = Path("udpipe_out/ancor")

stanza_out_dir.mkdir(parents=True, exist_ok=True)
udpipe_out_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


class ANCORDocument:
    def __init__(self, file_path):
        self.file_path = file_path
        self.tree = etree.parse(file_path)
        self.root = self.tree.getroot()

        self.words_dict: Dict[str, str]
        self.document: List[List[str]]
        self.words_ids: List[str]
        self.sentence_ids: List[str]
        self._parse_words()

        self.mentions_dict: Dict[str, Dict[str, Any]]
        self._parse_mentions()

        self.chains_dict: Dict[str, List[str]]
        self._parse_chains()

    def _parse_words(self):
        self.words_dict = {}
        self.words_ids = []
        self.document = []
        sentence = []
        self.sentence_ids = []
        prev_u = None
        prev_s = None
        for i, word in enumerate(self.tree.xpath(".//tei:w | .//tei:pc", namespaces=self.root.nsmap)):
            word_id = WordIndex.from_string(word.get("{http://www.w3.org/XML/1998/namespace}id"))
            s, u = word_id.s, word_id.u
            self.words_dict[word_id] = word.text
            self.words_ids.append(word_id)
            if i == 0:
                sentence.append(word.text)
            elif i > 0 and u != prev_u:
                if len(sentence) == 0:
                    logging.info(f"Empty sentence found at {s}.{u}")
                self.document.append(sentence)
                sentence = [word.text]
                self.sentence_ids.append(f"{prev_s}.{prev_u}")
            else:
                sentence.append(word.text)
            prev_u = u
            prev_s = s
        self.document.append(sentence)
        self.sentence_ids.append(f"{s}.{u}")

        assert len(self.document) == len(self.sentence_ids)

    def _parse_mentions(self):
        self.mentions_dict = {}
        for mention in self.tree.xpath('.//tei:spanGrp[@tei:subtype="mention"]/tei:span', namespaces=self.root.nsmap):
            self.mentions_dict[mention.get("{http://www.w3.org/XML/1998/namespace}id")] = {
                "continuous": True,
                "from": WordIndex.from_string(mention.get("{http://www.tei-c.org/ns/1.0}from")),
                "to": WordIndex.from_string(mention.get("{http://www.tei-c.org/ns/1.0}to")),
                "ana": mention.get("{http://www.tei-c.org/ns/1.0}ana")[1:],
            }

        for mention in self.tree.xpath('.//tei:spanGrp[@tei:subtype="expletive"]/tei:span', namespaces=self.root.nsmap):
            self.mentions_dict[mention.get("{http://www.w3.org/XML/1998/namespace}id")] = {
                "continuous": True,
                "from": WordIndex.from_string(mention.get("{http://www.tei-c.org/ns/1.0}from")),
                "to": WordIndex.from_string(mention.get("{http://www.tei-c.org/ns/1.0}to")),
                "ana": mention.get("{http://www.tei-c.org/ns/1.0}ana")[1:],
            }

        for mention in self.tree.xpath(
            './/tei:spanGrp[@tei:subtype="mention.discontinuous"]/tei:span', namespaces=self.root.nsmap
        ):
            m_words = [
                WordIndex.from_string(target)
                for target in mention.get("{http://www.tei-c.org/ns/1.0}target").split(" ")
            ]
            m_id = mention.get("{http://www.w3.org/XML/1998/namespace}id")

            # It seems that in the original dataset, the words are sorted using their string representations
            # when constructing discontinuous mentions. Thus, some of the discontinuous mentions are, in face, continuous.
            # For example, "#s19.u20.w10 #s19.u20.w11 #s19.u20.w9"
            # So, we will resort the words properly to ensure correct coversion.
            m_words = sorted(m_words, key=lambda x: x.w)

            # And then check if the sorted words make a continuous mention
            is_continuous = True
            for i in range(1, len(m_words)):
                if m_words[i].w - 1 != m_words[i - 1].w:
                    is_continuous = False
                    break

            if is_continuous:
                logging.info(
                    f"The mention with id {m_id} is marked as discontinuous but is actually continuous with the following word ids: {m_words}"
                )
                self.mentions_dict[m_id] = {
                    "continuous": is_continuous,
                    "from": m_words[0],
                    "to": m_words[-1],
                    "ana": mention.get("{http://www.tei-c.org/ns/1.0}ana")[1:],
                }
            else:
                overlapping = [k for k, v in self.mentions_dict.items() if "words" in v and v["words"][0] == m_words[0]]
                if len(overlapping) > 0:
                    logging.warning(f"Found a discontinuous mention {m_id} which is overlapping with an existing mentions {overlapping}!")
                    continue
                self.mentions_dict[m_id] = {
                    "continuous": is_continuous,
                    "words": m_words,
                    "ana": mention.get("{http://www.tei-c.org/ns/1.0}ana")[1:],
                }


    def _parse_chains(self):
        self.chains_dict = {}
        for chain in self.tree.xpath('.//tei:linkGrp[@tei:subtype="chain"]/tei:link', namespaces=self.root.nsmap):
            self.chains_dict[chain.get("{http://www.w3.org/XML/1998/namespace}id")] = [
                target[1:] for target in chain.get("{http://www.tei-c.org/ns/1.0}target").split(" ")
            ]
        for i, (mention_id, _) in enumerate(self.mentions_dict.items()):
            if "EXPLETIVE" in mention_id:
                self.chains_dict[f"s-EXPLETIVE-{i}"] = [mention_id]

        singletons = []
        in_chain = [m for k, chain in self.chains_dict.items() for m in chain]
        for mention in self.mentions_dict.keys():
            if mention not in in_chain:
                singletons.append(mention)

        for i, singleton in enumerate(singletons):
            self.chains_dict[f"s-SINGLETON-{i}"] = [singleton]


def create_ud_mentions(ud_doc, ancor_document, ancor2conll_ids):
    trees = list(ud_doc.trees)
    words = [tree.descendants for tree in trees]
    ents = []
    for chain_id, mention_ids in ancor_document.chains_dict.items():
        ent = ud_doc.create_coref_entity()
        for mention_idx in mention_ids:
            try:
                mention = ancor_document.mentions_dict[mention_idx]
            except KeyError as e:
                logging.error(f"Mention {mention_idx} was not found!")
                continue
            if mention["continuous"]:
                if mention["from"].u != mention["to"].u:
                    logging.warning(f"Found a continuous cross-sentence mention {mention_idx}, which is not supported!")
                    continue
                try:
                    start_id = ancor2conll_ids[mention["from"]]
                    end_id = ancor2conll_ids[mention["to"]]
                except KeyError as e:
                    logging.error(
                        f"Could not find the word {mention['from']} or {mention['to']} in {ancor_document.file_path}!"
                    )
                    logging.error(ancor2conll_ids)
                    # pprint(ancor2conll_ids)
                    raise e
                try:
                    m = ent.create_mention(words=words[start_id.sent_id][start_id.token_id - 1 : end_id.token_id])
                except IndexError as e:
                    logging.error(mention, start_id, end_id)
                    raise e
            else:
                # TODO: Add support for discontinuous mentions
                prev_u = mention["words"][0].u
                mention_words = []
                for i, mention_word in enumerate(mention["words"]):
                    u = mention_word.u
                    if i != 0 and prev_u != u:
                        logging.warning(
                            f"Found a discontinuous cross-sentence mention {mention_idx}, which is not supported!"
                        )
                        mention_words = []
                        break
                    try:
                        mention_words.append(
                            words[ancor2conll_ids[mention_word].sent_id][ancor2conll_ids[mention_word].token_id - 1]
                        )
                    except IndexError as e:
                        logging.warning(ancor_document.file_path, ancor2conll_ids[mention_word], mention_word)
                        raise e
                    prev_u = u
                if len(mention_words) > 0:
                    m = ent.create_mention(words=mention_words)
        if len(ent.mentions) > 0:
            ents.append(ent)


def sort_feats(ud_doc):
    # Somewhere along the processing pipeline (I guess in stanza), the FEATS are not properly sorted.
    # This causes the UD validation to fail.
    # I cannot find where exactly the bad sorting happens so the easiest solution for now it just to resort the FEATS post hoc.
    for node in ud_doc.nodes:
        node_feats = str(node.feats)
        sorted_feats = "|".join(sorted(node_feats.split("|"), key=lambda x: x.lower()))
        if node_feats != sorted_feats:
            logging.info(f"Re-sorted feats are different from the original ones: {node_feats} -> {sorted_feats}")
            node.feats = sorted_feats


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "corpus_name",
        type=str,
        choices=("corpus_ESLO", "corpus_ESLO_CO2", "corpus_OTG", "corpus_UBS"),
        help="The name of the corpus to process.",
    )

    args = argparser.parse_args()

    corpus_name = args.corpus_name
    corpus_path = Path(f"ortolang-000903/ortolang-000903/3/corpus/{corpus_name}")

    nlp_tokenized = stanza.Pipeline(lang="fr", processors="tokenize,mwt")
    nlp = stanza.Pipeline(lang="fr", processors="tokenize,pos,lemma,depparse", tokenize_pretokenized=True)

    for file_path in (pbar := tqdm(Path(data_path / corpus_path).glob("*.tei"))):
        with logging_redirect_tqdm():
            pbar.set_description(f"Processing {file_path}")

            ancor_document = ANCORDocument(file_path)

            doc = resplit_mwt(ancor_document.document, nlp_tokenized)
            doc = nlp(doc)

            ancor2conll_ids = {}
            total_word_count = 1
            for sent_id, (ancor_id, sentence) in enumerate(zip(ancor_document.sentence_ids, doc.sentences)):
                sentence.sent_id = f"{file_path.stem}.{ancor_id}"
                sentence.doc_id = file_path.stem
                for token_id, token in enumerate(sentence.tokens):
                    # We will need the mapping from the original word ids to the sentence and token ids in the CoNLL output later.
                    # `stanza` stores all the token ids in a tuple, so a single-word token will have an id like `(1,)`.
                    # and a multi-word token will have an id like `(1, 2)`.
                    # In French, the multi-word tokens are amalgams (ex. aux = à + les).
                    # In case the mention starts with a multi-word token in the original file, we don't want to include a preposition in it.
                    # Thus the last token id is taken every time.
                    ancor2conll_ids[ancor_document.words_ids[total_word_count - 1]] = (
                        CoNLLTokenLocation(sent_id=sent_id, token_id=token.id[-1])
                    )
                    total_word_count += 1

            CoNLL.write_doc2conll(doc, f"{stanza_out_dir}/{corpus_name}-{file_path.stem}.conllu")

            ud_doc = udapi.Document(f"{stanza_out_dir}/{corpus_name}-{file_path.stem}.conllu")

            create_ud_mentions(ud_doc, ancor_document, ancor2conll_ids)

            # Add the document name to the first sentence of the document
            ud_doc[0].trees[0].newdoc = f"{corpus_name}-{file_path.stem}"

            sort_feats(ud_doc)

            ud_doc.store_conllu(f"{udpipe_out_dir}/{corpus_name}-{file_path.stem}.conllu")


if __name__ == "__main__":
    main()
