import sys
import subprocess
import os
from shutil import copyfile
from collections import defaultdict

from read import read_UD_infile

from classifier.src.main_processor import GeneralAnalyzer


def get_config_by_genre(genre):
    if genre == "historic":
        return "17cent"
    elif genre == "poetry":
        return "poetry"
    elif genre == "social":
        return "social"
    return "other"

FOLDER = "tmp"
os.makedirs(FOLDER, exist_ok=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Pass the input and output files.")
    input_file, output_file = sys.argv[1:]
    cls = GeneralAnalyzer(static_dir="classifier/static")
    list_of_sentences = cls.parse_conllu_file(input_file)
    texts = cls._get_str_of_text(list_of_sentences)
    predicted_genres = cls._classify_texts(texts)

    input_data = read_UD_infile(input_file)
    input_sents = [" ".join(elem[1] for elem in sent) for sent in input_data]
    ids_by_genres = defaultdict(list)
    for i, genre in enumerate(predicted_genres):
        ids_by_genres[genre].append(i)
    for genre, indexes in ids_by_genres.items():
        with open("tmp/{}-source.conllu".format(genre), "w", encoding="utf8") as fout:
            for i in indexes:
                fout.write("\n".join(elem[1] for elem in input_data[i]) + "\n\n")

    for genre, indexes in ids_by_genres.items():
        print(genre, end=" ")
        infile = "{}/{}-source.conllu".format(FOLDER, genre)
        with open(infile, "w", encoding="utf8") as fout:
            for i in indexes:
                fout.write("\n".join(elem[1] for elem in input_data[i]) + "\n\n")
        outfile = os.path.join(FOLDER, "{}-result.conllu".format(genre))
        config_name = get_config_by_genre(genre)
        config = "config/{}.json".format(config_name)
        with open(infile, "r", encoding="utf8") as fin, open(outfile, "w", encoding="utf8") as fout:
            subprocess.run(["python", "-m", "deeppavlov.models.morpho_tagger", 
                            config, "-i", "vertical", "-o", "ud", "-b", "8"], stdin=fin, stdout=fout)
    
    answer = [None] * len(input_data)
    for genre, indexes in ids_by_genres.items():
        outfile = os.path.join(FOLDER, "{}-result.conllu".format(genre))
        curr_sents = read_UD_infile(outfile)
        assert len(curr_sents) == len(indexes)
        for i, index in enumerate(indexes):
            answer[index] = curr_sents[i]
    with open(output_file, "w", encoding="utf8") as fout:
        for sent in answer:
            for i, elem in enumerate(sent):
                if len(elem[5]) > 0:
                    elem[5] = "|".join("{}={}".format(*elem) for elem in sorted(elem[5].items()))
                else:
                    elem[5] = "_"
            fout.write("\n".join("\t".join(map(str, elem)) for elem in sent) + "\n\n")
    