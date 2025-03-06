import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import stanza
import udapi
from lxml import etree
from udapi.core.coref import BridgingLinks, CorefMention
from stanza.models.mwt.utils import resplit_mwt
from stanza.utils.conll import CoNLL
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ancor_patch import apply_patch
from ancor_utils import CoNLLTokenLocation, WordIndex

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
        apply_patch(self.mentions_dict)

        self.chains_dict: Dict[str, List[str]]
        self._parse_chains()

        self.bridging_dict: Dict[str, Dict[str, Any]]
        self._parse_bridging()

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
            # when constructing discontinuous mentions. Thus, some of the discontinuous mentions are, in fact, continuous.
            # For example, "#s19.u20.w10 #s19.u20.w11 #s19.u20.w9"
            # So, we will resort the words properly to ensure correct coversion.
            m_words = sorted(m_words, key=lambda x: (x.u, x.w))

            # And then check if the sorted words make a continuous mention
            is_continuous = True
            is_cross_sentence = False
            d_spans = []
            d_span = [m_words[0]]
            for i in range(1, len(m_words)):
                if m_words[i].w - 1 != m_words[i - 1].w:
                    is_continuous = False
                    d_spans.append(d_span + [m_words[i - 1]])
                    d_span = [m_words[i]]
                if m_words[i].u != m_words[i - 1].u:
                    is_cross_sentence = True
                    logging.info(f"Found cross-sentence mention {m_words} in {self.file_path}")
            d_spans.append(d_span + [m_words[-1]])

            if is_continuous:
                logging.info(
                    f"The mention with id {m_id} is marked as discontinuous but is actually continuous with the following word ids: {m_words}"
                )
                self.mentions_dict[m_id] = {
                    "continuous": is_continuous,
                    "cross-sentence": is_cross_sentence,
                    "from": m_words[0],
                    "to": m_words[-1],
                    "ana": mention.get("{http://www.tei-c.org/ns/1.0}ana")[1:],
                }
            else:
                self.mentions_dict[m_id] = {
                    "continuous": is_continuous,
                    "cross-sentence": is_cross_sentence,
                    "words": d_spans,
                    "ana": mention.get("{http://www.tei-c.org/ns/1.0}ana")[1:],
                }

    def _parse_chains(self):
        self.chains_dict = {}
        for chain in self.tree.xpath('.//tei:linkGrp[@tei:subtype="chain"]/tei:link', namespaces=self.root.nsmap):
            chain_mentions = [target[1:] for target in chain.get("{http://www.tei-c.org/ns/1.0}target").split(" ")]
            overlapping = []
            for i in range(len(chain_mentions) - 1):
                for j in range(i + 1, len(chain_mentions)):
                    if (
                        not self.mentions_dict[chain_mentions[i]]["continuous"]
                        and not self.mentions_dict[chain_mentions[j]]["continuous"]
                        and self.mentions_dict[chain_mentions[i]]["words"][0]
                        == self.mentions_dict[chain_mentions[j]]["words"][0]
                    ):
                        logging.warning(
                            f"Found two overlapping discontinuous mentions {chain_mentions[i]} and {chain_mentions[j]} in the same corerefence chain! The second one ({chain_mentions[i]}) will be removed."
                        )
                        overlapping.append(j)
            for o in sorted(overlapping, reverse=True):
                del chain_mentions[o]
            self.chains_dict[chain.get("{http://www.w3.org/XML/1998/namespace}id")] = chain_mentions
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

    def _parse_bridging(self):
        self.bridging_dict = {}
        for coref in self.tree.xpath(
            './/tei:linkGrp[@tei:subtype="associative_anaphora"]/tei:link', namespaces=self.root.nsmap
        ):
            self.bridging_dict[coref.get("{http://www.w3.org/XML/1998/namespace}id")] = {
                "target": [target[1:] for target in coref.get("{http://www.tei-c.org/ns/1.0}target").split(" ")],
                "ana": coref.get("{http://www.tei-c.org/ns/1.0}ana")[1:],
            }


def create_ud_mentions(ud_doc, ancor_document, ancor2conll_ids):
    trees = list(ud_doc.trees)
    words = [tree.descendants for tree in trees]
    ents = []
    mention_to_ent = {}
    for chain_id, mention_ids in ancor_document.chains_dict.items():
        ent = ud_doc.create_coref_entity()
        for mention_idx in mention_ids:
            m: Optional[CorefMention] = None
            try:
                mention = ancor_document.mentions_dict[mention_idx]
            except KeyError:
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
            elif not mention["continuous"] and mention["cross-sentence"]:
                logging.warning(f"Found a discontinuous cross-sentence mention {mention_idx}, which is not supported!")
            else:
                mention_words = []
                for i, word_span in enumerate(mention["words"]):
                    start_id = ancor2conll_ids[word_span[0]]
                    end_id = ancor2conll_ids[word_span[-1]]
                    mention_words += words[start_id.sent_id][start_id.token_id - 1 : end_id.token_id]
                    # prev_u = u
                if len(mention_words) > 0:
                    m = ent.create_mention(words=mention_words)
            if m:
                mention_to_ent[mention_idx] = {"eid": ent.eid, "mention": m}
        if len(ent.mentions) > 0:
            ents.append(ent)

    bridging_links = {}
    for bridge_idx, bridge in ancor_document.bridging_dict.items():
        tgt, src = bridge["target"]
        rel_tei_path = f'.//tei:fs[@xml:id="{bridge["ana"]}"]/tei:f[@tei:name="type"]/tei:string'
        rel_element = ancor_document.tree.xpath(rel_tei_path, namespaces=ancor_document.root.nsmap)
        rel_type = ":pronominal" if rel_element[0].text == "ASSOC_PRONOM" else ""
        try:
            if src in bridging_links:
                bridging_links[src] += f",{mention_to_ent[tgt]['eid']}<{mention_to_ent[src]['eid']}{rel_type}"
            else:
                bridging_links[src] = f"{mention_to_ent[tgt]['eid']}<{mention_to_ent[src]['eid']}{rel_type}"
        except KeyError:
            logging.error(f"Tried to create a bridging link from {tgt} to {src} but a mention was missing, skipping...")

    bls = []
    for src_idx, b_string in bridging_links.items():
        try:
            bls.append(BridgingLinks.from_string(b_string, ud_doc.eid_to_entity, ud_doc))
        except ValueError:
            logging.error("Something happened while adding a bridging link, skipping...")


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
                sentence.sent_id = f"{corpus_name}-{file_path.stem}.{ancor_id}"
                for token_id, token in enumerate(sentence.tokens):
                    # We will need the mapping from the original word ids to the sentence and token ids in the CoNLL output later.
                    # `stanza` stores all the token ids in a tuple, so a single-word token will have an id like `(1,)`.
                    # and a multi-word token will have an id like `(1, 2)`.
                    # In French, the multi-word tokens are amalgams (ex. aux = à + les).
                    # In case the mention starts with a multi-word token in the original file, we don't want to include a preposition in it.
                    # Thus the last token id is taken every time.
                    ancor2conll_ids[ancor_document.words_ids[total_word_count - 1]] = CoNLLTokenLocation(
                        sent_id=sent_id, token_id=token.id[-1]
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
