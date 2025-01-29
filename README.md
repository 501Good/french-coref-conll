# French Coref CoNLL

Scripts for converting existing coreference corpora in French to the CoNLL-U compatible format.

This repository contains scripts for conferting ANCOR corpus (Muzerelle et al., 2014) and the French part of ParCorFull 2.0 corpus (Lapshinova-Koltunski et al., 2022) into the [CorefUD](https://ufal.mff.cuni.cz/corefud) format.

## How to Use

### Requirements

The main requirement is [uv](https://github.com/astral-sh/uv), an extremely fast Python package and project manager.

### Data

First, you will need to download the original data:
- ANCOR: https://www.ortolang.fr/market/corpora/ortolang-000903
- ParCorFull 2.0: https://github.com/chardmeier/parcor-full

For ANCOR, just click the green "Télécharger" button and extract the archive into the `data/` folder.

For ParCorFull 2.0, just `git clone` the repository into the `data/` folder.

In the end, your `data/` folder should have the following structure:

```
data
├── ortolang-000903
│   └── ortolang-000903
│       └── 3
│           └── corpus
└── parcor-full
    ├── corpus
    │   ├── DiscoMT
    │   │   ├── DE
    │   │   ├── EN
    │   │   └── FR
    │   ├── news
    │   │   ├── DE
    │   │   ├── EN
    │   │   └── PT
    │   └── TED
    │       ├── DE
    │       ├── EN
    │       ├── FR
    │       └── PT
    ...
```

### Run the convertion scripts

To convert the ANCOR corpus, run

`uv run bash convert_ancor_to_corefud.sh`

To convert the ParCorFull 2.0 corpus, run 

`uv run bash convert_parcorfull_to_corefud.sh`

The converted corpora will be placed into the `CorefUD/` folder.

## Current Roadmap

- [x] Convert the corpora to a valid CorefUD format
- [x] Expand multi-word tokens
- [x] Add generated morpho-syntactic information
- [ ] Process bridging anaphora in ANCOR (currently it is treated as a normal anaphora)
- [ ] Check how split antecedents are processed
- [ ] Process entities that span over several sentences
- [ ] In ANCOR, there are discontinuous entities that have the same parts, which currently breaks the format

## References

1. Lapshinova-Koltunski, Ekaterina et al. (juin 2022). “ParCorFull2.0: a Parallel Corpus Annotated with Full Coreference”. In Proceedings of the Thirteenth Language Resources and Evaluation Conference. Sous la dir. de Nicoletta Calzolari et al. Marseille, France, European Language Resources Association, p. 805-813. url: https://aclanthology.org/2022.lrec-1.85/.

2. Muzerelle, Judith et al. (2014). “ANCOR Centre, a large free spoken French coreference corpus: description of the resource and reliability measures”. In LREC’2014, 9th Language Resources and Evaluation Conference. P. 843-847.