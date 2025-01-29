import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

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


data_path = Path("data")

stanza_out_dir = Path("stanza_out/ancor")
udpipe_out_dir = Path("udpipe_out/ancor")

stanza_out_dir.mkdir(parents=True, exist_ok=True)
udpipe_out_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


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
            tree = etree.parse(file_path)
            root = tree.getroot()

            words_dict = {}
            words_ids = []
            document = []
            sentence = []
            sentence_ids = []
            prev_u = None
            prev_s = None
            for i, word in enumerate(tree.xpath(".//tei:w | .//tei:pc", namespaces=root.nsmap)):
                word_id = word.get("{http://www.w3.org/XML/1998/namespace}id")
                words_dict[word_id] = word.text
                words_ids.append(word_id)
                try:
                    if ".dash" in word_id:
                        s, u, _, _ = word_id.split(".")
                    else:
                        s, u, _ = word_id.split(".")
                except ValueError as e:
                    logging.error(f"Tried to split '{word_id}' but failed! Perhaps it is badly formatted?")
                    raise e
                if i == 0:
                    sentence.append(word.text)
                elif i > 0 and u != prev_u:
                    if len(sentence) == 0:
                        logging.info(f"Empty sentence found at {s}.{u}")
                    document.append(sentence)
                    sentence = [word.text]
                    sentence_ids.append(f"{prev_s}.{prev_u}")
                else:
                    sentence.append(word.text)
                prev_u = u
                prev_s = s
            document.append(sentence)
            sentence_ids.append(f"{s}.{u}")

            assert len(document) == len(sentence_ids)

            doc = resplit_mwt(document, nlp_tokenized)
            doc = nlp(doc)

            mentions_dict = {}
            for mention in tree.xpath('.//tei:spanGrp[@tei:subtype="mention"]/tei:span', namespaces=root.nsmap):
                mentions_dict[mention.get("{http://www.w3.org/XML/1998/namespace}id")] = {
                    "continuous": True,
                    "from": mention.get("{http://www.tei-c.org/ns/1.0}from")[1:],
                    "to": mention.get("{http://www.tei-c.org/ns/1.0}to")[1:],
                    "ana": mention.get("{http://www.tei-c.org/ns/1.0}ana")[1:],
                }

            for mention in tree.xpath('.//tei:spanGrp[@tei:subtype="expletive"]/tei:span', namespaces=root.nsmap):
                mentions_dict[mention.get("{http://www.w3.org/XML/1998/namespace}id")] = {
                    "continuous": True,
                    "from": mention.get("{http://www.tei-c.org/ns/1.0}from")[1:],
                    "to": mention.get("{http://www.tei-c.org/ns/1.0}to")[1:],
                    "ana": mention.get("{http://www.tei-c.org/ns/1.0}ana")[1:],
                }

            for mention in tree.xpath(
                './/tei:spanGrp[@tei:subtype="mention.discontinuous"]/tei:span', namespaces=root.nsmap
            ):
                m_words = [target[1:] for target in mention.get("{http://www.tei-c.org/ns/1.0}target").split(" ")]
                m_id = mention.get("{http://www.w3.org/XML/1998/namespace}id")
                if len([1 for v in mentions_dict.values() if "words" in v and v["words"][0] == m_words[0]]) > 0:
                    logging.warning(f"Found a discontinuous mention {m_id} which is overlapping with an existing mention!")
                else:
                    mentions_dict[m_id] = {
                        "continuous": False,
                        "words": m_words,
                        "ana": mention.get("{http://www.tei-c.org/ns/1.0}ana")[1:],
                    }

            corefs_dict = {}
            for coref in tree.xpath('.//tei:linkGrp[@tei:subtype="coreference"]/tei:link', namespaces=root.nsmap):
                corefs_dict[coref.get("{http://www.w3.org/XML/1998/namespace}id")] = {
                    "target": [target[1:] for target in coref.get("{http://www.tei-c.org/ns/1.0}target").split(" ")],
                    "ana": mention.get("{http://www.tei-c.org/ns/1.0}ana")[1:],
                }

            chains_dict = {}
            for chain in tree.xpath('.//tei:linkGrp[@tei:subtype="chain"]/tei:link', namespaces=root.nsmap):
                chains_dict[chain.get("{http://www.w3.org/XML/1998/namespace}id")] = [
                    target[1:] for target in chain.get("{http://www.tei-c.org/ns/1.0}target").split(" ")
                ]
            for i, (mention_id, _) in enumerate(mentions_dict.items()):
                if "EXPLETIVE" in mention_id:
                    chains_dict[f"s-EXPLETIVE-{i}"] = [mention_id]

            singletons = []
            in_chain = [m for k, chain in chains_dict.items() for m in chain if "EXPLETIVE" not in m]
            for mention in mentions_dict.keys():
                if mention not in in_chain:
                    singletons.append(mention)

            for i, singleton in enumerate(singletons):
                chains_dict[f"s-SINGLETON-{i}"] = [singleton]

            mention_groups = {}
            for k, v in corefs_dict.items():
                if v["target"][0] in mention_groups:
                    mention_groups[v["target"][0]].append(v["target"][1])
                else:
                    mention_groups[v["target"][0]] = [v["target"][1]]

            ancor2conll_ids = {}
            total_word_count = 1
            for sent_id, (ancor_id, sentence) in enumerate(zip(sentence_ids, doc.sentences)):
                sentence.sent_id = f"{file_path.stem}.{ancor_id}"
                sentence.doc_id = file_path.stem
                for token_id, token in enumerate(sentence.tokens):
                    # We will need the mapping from the original word ids to the sentence and token ids in the CoNLL output later.
                    # `stanza` stores all the token ids in a tuple, so a single-word token will have an id like `(1,)`.
                    # and a multi-word token will have an id like `(1, 2)`.
                    # In French, the multi-word tokens are amalgams (ex. aux = à + les).
                    # In case the mention starts with a multi-word token in the original file, we don't want to include a preposition in it.
                    # Thus the last token id is taken every time.
                    ancor2conll_ids[words_ids[total_word_count - 1]] = CoNLLTokenLocation(
                        sent_id=sent_id, token_id=token.id[-1]
                    )
                    total_word_count += 1

            CoNLL.write_doc2conll(doc, f"{stanza_out_dir}/{corpus_name}-{file_path.stem}.conllu")

            ud_doc = udapi.Document(f"{stanza_out_dir}/{corpus_name}-{file_path.stem}.conllu")

            trees = list(ud_doc.trees)
            words = [tree.descendants for tree in trees]
            ents = []
            for chain_id, mention_ids in chains_dict.items():
                ent = ud_doc.create_coref_entity()
                for mention_idx in mention_ids:
                    try:
                        mention = mentions_dict[mention_idx]
                    except KeyError as e:
                        logging.error(f"Mention {mention_idx} was not found!")
                        continue
                    if mention["continuous"]:
                        if mention["from"].split(".")[1] != mention["to"].split(".")[1]:
                            logging.warning(f"Found a continuous multi-sentence mention {mention_idx}, which is not supported!")
                            continue
                        try:
                            start_id = ancor2conll_ids[mention["from"]]
                            end_id = ancor2conll_ids[mention["to"]]
                        except KeyError as e:
                            logging.error(f"Could not find the word {mention['from']} or {mention['to']} in {file_path}!")
                            # pprint(ancor2conll_ids)
                            raise e
                        try:
                            m = ent.create_mention(words=words[start_id.sent_id][start_id.token_id - 1 : end_id.token_id])
                        except IndexError as e:
                            logging.error(mention, start_id, end_id)
                            raise e
                    else:
                        # TODO: Add support for discontinuous mentions
                        prev_u = mention["words"][0].split(".")[1]
                        mention_words = []
                        for i, mention_word in enumerate(mention["words"]):
                            u = mention_word.split(".")[1]
                            if i != 0 and prev_u != u:
                                logging.warning(
                                    f"Found a discontinuous multi-sentence mention {mention_idx}, which is not supported!"
                                )
                                mention_words = []
                                break
                            try:
                                mention_words.append(
                                    words[ancor2conll_ids[mention_word].sent_id][ancor2conll_ids[mention_word].token_id - 1]
                                )
                            except IndexError as e:
                                logging.warning(file_path, ancor2conll_ids[mention_word], mention_word)
                                raise e
                            prev_u = u
                        if len(mention_words) > 0:
                            m = ent.create_mention(words=mention_words)
                if len(ent.mentions) > 0:
                    ents.append(ent)

            # Add the document name to the first sentence of the document
            ud_doc[0].trees[0].newdoc = f"{corpus_name}-{file_path.stem}"

            # Somewhere along the processing pipeline (I guess in stanza), the FEATS are not properly sorted.
            # This causes the UD validation to fail.
            # I cannot find where exactly the bad sorting happens so the easiest solution for now it just to resort the FEATS post hoc.
            for node in ud_doc.nodes:
                node_feats = str(node.feats)
                sorted_feats = "|".join(sorted(node_feats.split("|"), key=lambda x: x.lower()))
                if node_feats != sorted_feats:
                    logging.info(f"Re-sorted feats are different from the original ones: {node_feats} -> {sorted_feats}")
                    node.feats = sorted_feats

            # Print the newly created coreference entities.
            # udapi.create_block("corefud.PrintEntities")[1].process_document(ud_doc)

            ud_doc.store_conllu(f"{udpipe_out_dir}/{corpus_name}-{file_path.stem}.conllu")


if __name__ == "__main__":
    main()
