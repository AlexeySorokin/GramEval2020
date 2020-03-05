import string

NO_SPACE = 'SpaceAfter=No'
CUSTOM_PUNCTUATION_SET = set(string.punctuation + '`' + '—')

def conll_to_txt_with_space_after(list_with_words):
	to_print = ''
	prev_comment = ''
	for word in list_with_words[:-1]:  # под -1 лежит extra_space
		token = word[1]
		comment = word[-1]
		if token == '``' or token == '&#39;&#39;':
			token = '"'
		elif token == '--':
			token = '—'	
		if prev_comment == NO_SPACE or to_print == '':
			to_print += token
		else:
			to_print += ' ' 
			to_print += token
		prev_comment = comment
	return to_print

def conll_to_txt_without_space_after(list_with_words):
	to_print = ''
	has_text_for_sent = False
	skip_space = False
	has_opening_quotation = False
	for word in list_with_words[:-1]:  # под -1 лежит extra_space
		token = word[1]  
		if to_print == '':
			if token == '``':
				to_print += '"'
				skip_space = True
				has_opening_quotation = True
			else:
				to_print += token
		elif skip_space:
			if token == '``':
				to_print += '"'
				skip_space = True
				has_opening_quotation = True
			else:
				to_print += token
				skip_space = False
		elif token == '&#39;&#39;':
			to_print += '"'
		elif set(token).issubset(CUSTOM_PUNCTUATION_SET):
			if token == '``':
				to_print += ' "'
				skip_space = True
				has_opening_quotation = True
			elif token == '"' or token == '\'\'':
				if has_opening_quotation:
					to_print += '"'
					has_opening_quotation = False
				else:
					to_print += ' "'
					skip_space = True 
					has_opening_quotation = True
			elif token == '--' or token == '—':
				to_print += ' —'
			elif token == '-':
				to_print += '-'
				skip_space = True
			else:
				to_print += token
		else:
			to_print += ' ' 
			to_print += token
	return to_print
