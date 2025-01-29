corpusNames=('TED' 'DiscoMT')

for name in "${corpusNames[@]}"; do
    python src/mmax2conll.py "$name"
done

udapy read.Conllu files='!./udpipe_out/parcorfull/*.conllu' corefud.MergeSameSpan corefud.IndexClusters corefud.MoveHead write.Conllu > fr_parcorfull-corefud.conllu
udapy read.Conllu files='fr_parcorfull-corefud.conllu' corefud.FixInterleaved corefud.FixToValidate write.Conllu overwrite=1
python src/split_conll.py ParCorFull