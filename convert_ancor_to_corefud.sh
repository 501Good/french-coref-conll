corpusNames=('corpus_ESLO' 'corpus_ESLO_CO2' 'corpus_OTG' 'corpus_UBS')

for name in "${corpusNames[@]}"; do
    python src/ancor2conll.py "$name"
done

udapy read.Conllu files='!./udpipe_out/ancor/*.conllu' corefud.MergeSameSpan corefud.IndexClusters corefud.MoveHead write.Conllu > fr_ancor-corefud.conllu
udapy read.Conllu files='fr_ancor-corefud.conllu' corefud.FixInterleaved corefud.FixToValidate write.Conllu overwrite=1
python src/split_conll.py ANCOR