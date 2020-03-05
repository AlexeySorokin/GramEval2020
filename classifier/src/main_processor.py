#import sys
#sys.path.append('/usr/local/lib/python3.7/site-packages')
import os
import json
import pickle
import re

import numpy as np
from pymorphy2 import MorphAnalyzer
# from rnnmorph.predictor import RNNMorphPredictor
from russian_tagsets import converters
from tqdm import tqdm

from classifier.src.conllu_to_txt import conll_to_txt_with_space_after, conll_to_txt_without_space_after

class GeneralAnalyzer:

	def __init__(self, static_dir="../static", vectorizer="charwb_vectorizer", classifier="genre_classifier"):
		# self.morph_predictor = RNNMorphPredictor(language='ru')
		self.secondary_analyzer = MorphAnalyzer()
		with open(os.path.join(static_dir, "{}.pkl".format(vectorizer)), "rb") as inf:
			self.vectorizer = pickle.load(inf)
		with open(os.path.join(static_dir, "{}.pkl".format(classifier)), "rb") as inf:
			self.genre_classifier = pickle.load(inf)
		with open(os.path.join(static_dir, "lists.json")) as inf:
			self.dicts = json.load(inf)
		self.latin = re.compile("^[A-Za-z0-9]+$")
		self.digit = re.compile("^[0-9]+$")
		self.pm_to_ud = converters.converter('opencorpora-int', 'ud20')

	def parse_conllu_file(self, inf):
		what_to_extract = []
		with open(inf) as f:
			sentences = f.read().split("\n\n")
		for i, sentence in enumerate(sentences):
			has_extra_space = False
			if sentence:
				sent = []
				words = sentence.split("\n")
				for word in words:
					if word.startswith("#"):
						continue
					elif "\t" not in word:
						has_extra_space=True
						continue
					info = word.split("\t")
					idx = info[0]
					text = info[1]
					sent.append([idx, text])
				sent.append(has_extra_space)
				what_to_extract.append(sent)
		return what_to_extract

	def write_to_conllu(self, list_of_sentences):
		with open("../static/GramEval_private_test.conllu", "w") as out:
			for i, sentence in enumerate(list_of_sentences):
				has_extra_space = sentence[-1]
				if has_extra_space and i != len(list_of_sentences)-1:
					out.write("\n")
				for word in sentence[:-1]: # -1 -- признак, есть ли лишний пробел
					out.write("\t".join(word))
					out.write("\n")
				out.write("\n")
		return 1

	def _get_str_of_text(self, texts, spaces_after=False):
		list_of_str_texts = []
		for text in texts:
			if spaces_after:
				list_of_str_texts.append(conll_to_txt_with_space_after(text))
			else:
				list_of_str_texts.append(conll_to_txt_without_space_after(text))
		return list_of_str_texts

	def _classify_texts(self, texts):
		vectors = self.vectorizer.transform(texts)
		genres = self.genre_classifier.predict(vectors)
		return genres

	def _preprocess_word(self, word):
		new_word = ""
		if word[-1] in self.dicts["ending_letters"] and len(word) > 1 and word.lower() not in self.dicts["exceptions"] and not word.endswith('ать'):
			word = word[:-1]
		for letter in word:
			new_word += self.dicts["to_be_replaced"].get(letter, letter)
		new_word = self.dicts["word_replacements"].get(new_word.lower(), new_word)
		return new_word

	def _change_rules(self, word, true_pos, true_lemma, enriched_anal, historic=False):
		lowered_word = word.lower()
		if lowered_word in self.dicts["straight_lemma_mapping"]:
			true_lemma = self.dicts["straight_lemma_mapping"][lowered_word]
		elif true_lemma in self.dicts["conditional_lemma_mapping"]:
			for key in self.dicts["conditional_lemma_mapping"][true_lemma]:
				if key in enriched_anal:
					true_lemma = self.dicts["conditional_lemma_mapping"][true_lemma][key]
					break
		if lowered_word in self.dicts["direct_pos_mapping"]:
			true_pos = self.dicts["direct_pos_mapping"][lowered_word]
		if historic and lowered_word in self.dicts["direct_pos_mapping_17th"]:
			true_pos = self.dicts["direct_pos_mapping_17th"][lowered_word]
		if re.match(self.digit, word):
			true_pos = "NUM"
			true_lemma = lowered_word
		elif re.match(self.latin, word):
			true_pos = "X"
			true_lemma = lowered_word
			enriched_anal = "Foreign=Yes"
		if historic and lowered_word in self.dicts["enriched_anal_17th"]:
			enriched_anal = self.dicts["enriched_anal_17th"][lowered_word]
		return true_pos, true_lemma, enriched_anal

	def choose_pymorphy_form(self, word, true_pos, true_lemma, enriched_anal):
		hypotheses = self.secondary_analyzer.parse(word)
		hyp = None
		for hyp in hypotheses:
			if hyp.normal_form == true_lemma:
				break
		if not hyp:
			return true_pos, true_lemma, enriched_anal
		str_tag = str(hyp.tag)
		if hyp.tag.animacy:
			enriched_anal += self.dicts["tags_mapping"][hyp.tag.animacy]
		if hyp.tag.aspect:
			enriched_anal += self.dicts["tags_mapping"][hyp.tag.aspect]
		if "Surn" in str_tag or "Name" in str_tag or "Patr" in str_tag or "Geo" in str_tag:
			true_pos = "PROPN"
		elif true_pos == "ADJ" and "Degree=Pos" in enriched_anal:
			hyp = hypotheses[0]
			if hyp.tag.POS == "VERB":
				true_pos = "VERB"
				true_lemma = hyp.normal_form
				enriched_anal = self.pm_to_ud(str_tag).split(" ")[1]
		return true_pos, true_lemma, enriched_anal

	def __call__(self, conll_file):
		all_sentences = []
		list_of_sentences = self.parse_conllu_file(conll_file)
		texts = self._get_str_of_text(list_of_sentences)
		predicted_genres = self._classify_texts(texts)
		for i, sentence in enumerate(tqdm(list_of_sentences)):
			has_extra_space = sentence[-1]
			historic = predicted_genres[i] == "historic"
			rnnmorph_input = []
			rnn_s = []
			for text_tuple in sentence[:-1]: # потому что последним лежит лишний пробел
				word = text_tuple[1]
				new_word = self._preprocess_word(word) if historic else word
				rnnmorph_input.append(new_word)
			rnnmorph_output = self.morph_predictor.predict(rnnmorph_input)
			for j, word in enumerate(rnnmorph_output):
				pos = word.pos
				lemma = word.normal_form
				tag = word.tag
				if pos in {"NOUN", "VERB", "ADJ"}:
					pos, lemma, tag = self.choose_pymorphy_form(word.word, pos, lemma, tag)
				true_pos, true_lemma, true_tag = self._change_rules(word.word, pos, lemma, tag, historic=historic)
				if sentence[j][1][-1] == "ъ":
					true_lemma += "ъ"
				if historic and len(true_lemma) > 2 and true_lemma[-2:] == "ть":
					true_lemma = true_lemma[:-2] + "ти"
				final_list = [str(j+1), sentence[j][1], true_lemma, true_pos, "_", true_tag, "_", "_", "_", "_"]
				rnn_s.append(final_list)
			rnn_s.append(has_extra_space)
			all_sentences.append(rnn_s)
		self.write_to_conllu(all_sentences)
		return 1

if __name__ == '__main__':
	ga = GeneralAnalyzer()
	ga("../static/GramEval_private_test_clear.conllu")

