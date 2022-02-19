import os
import pandas as pd
import string
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
# Importing Libraries 
import nltk
import re
from unidecode import unidecode
nltk.download('stopwords') 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords 


class DPM_preprocessing:

	def __init__(self, train_path, test_path):

		self.train_path = train_path
		self.test_path = test_path
		self.train_task1_df = None
		self.train_task2_df = None
		self._80_negative_df = None
		self._20_negative_df = None
		self._80_positive_df = None
		self._20_positive_df = None
		self.test_set = None
		self.CONTRACTION_MAP = {
		"ain't": "is not",
		"aren't": "are not",
		"can't": "cannot",
		"can't've": "cannot have",
		"'cause": "because",
		"could've": "could have",
		"couldn't": "could not",
		"couldn't've": "could not have",
		"didn't": "did not",
		"doesn't": "does not",
		"don't": "do not",
		"hadn't": "had not",
		"hadn't've": "had not have",
		"hasn't": "has not",
		"haven't": "have not",
		"he'd": "he would",
		"he'd've": "he would have",
		"he'll": "he will",
		"he'll've": "he he will have",
		"he's": "he is",
		"how'd": "how did",
		"how'd'y": "how do you",
		"how'll": "how will",
		"how's": "how is",
		"i'd": "i would",
		"i'd've": "i would have",
		"i'll": "i will",
		"i'll've": "i will have",
		"i'm": "i am",
		"i've": "i have",
		"isn't": "is not",
		"it'd": "it would",
		"it'd've": "it would have",
		"it'll": "it will",
		"it'll've": "it will have",
		"it's": "it is",
		"let's": "let us",
		"ma'am": "madam",
		"mayn't": "may not",
		"might've": "might have",
		"mightn't": "might not",
		"mightn't've": "might not have",
		"must've": "must have",
		"mustn't": "must not",
		"mustn't've": "must not have",
		"needn't": "need not",
		"needn't've": "need not have",
		"o'clock": "of the clock",
		"oughtn't": "ought not",
		"oughtn't've": "ought not have",
		"shan't": "shall not",
		"sha'n't": "shall not",
		"shan't've": "shall not have",
		"she'd": "she would",
		"she'd've": "she would have",
		"she'll": "she will",
		"she'll've": "she will have",
		"she's": "she is",
		"should've": "should have",
		"shouldn't": "should not",
		"shouldn't've": "should not have",
		"so've": "so have",
		"so's": "so as",
		"that'd": "that would",
		"that'd've": "that would have",
		"that's": "that is",
		"there'd": "there would",
		"there'd've": "there would have",
		"there's": "there is",
		"they'd": "they would",
		"they'd've": "they would have",
		"they'll": "they will",
		"they'll've": "they will have",
		"they're": "they are",
		"they've": "they have",
		"to've": "to have",
		"wasn't": "was not",
		"we'd": "we would",
		"we'd've": "we would have",
		"we'll": "we will",
		"we'll've": "we will have",
		"we're": "we are",
		"we've": "we have",
		"weren't": "were not",
		"what'll": "what will",
		"what'll've": "what will have",
		"what're": "what are",
		"what's": "what is",
		"what've": "what have",
		"when's": "when is",
		"when've": "when have",
		"where'd": "where did",
		"where's": "where is",
		"where've": "where have",
		"who'll": "who will",
		"who'll've": "who will have",
		"who's": "who is",
		"who've": "who have",
		"why's": "why is",
		"why've": "why have",
		"will've": "will have",
		"won't": "will not",
		"won't've": "will not have",
		"would've": "would have",
		"wouldn't": "would not",
		"wouldn't've": "would not have",
		"y'all": "you all",
		"y'all'd": "you all would",
		"y'all'd've": "you all would have",
		"y'all're": "you all are",
		"y'all've": "you all have",
		"you'd": "you would",
		"you'd've": "you would have",
		"you'll": "you will",
		"you'll've": "you will have",
		"you're": "you are",
		"you've": "you have",
		}
	def remove_newlines_tabs(self, text):
		"""
		This function will remove all the occurrences of newlines, tabs, and combinations like: \\n, \\.
		
		arguments:
			input_text: "text" of type "String". 
						
		return:
			value: "text" after removal of newlines, tabs, \\n, \\ characters.
			
		Example:
		Input : This is her \\ first day at this place.\n Please,\t Be nice to her.\\n
		Output : This is her first day at this place. Please, Be nice to her. 
		
		"""
		
		# Replacing all the occurrences of \n,\\n,\t,\\ with a space.
		Formatted_text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('. com', '.com')
		return Formatted_text
	def strip_html_tags(self, text):
		""" 
		This function will remove all the occurrences of html tags from the text.
		
		arguments:
			input_text: "text" of type "String". 
						
		return:
			value: "text" after removal of html tags.
			
		Example:
		Input : This is a nice place to live. <IMG>
		Output : This is a nice place to live.  
		"""
		# Initiating BeautifulSoup object soup.
		soup = BeautifulSoup(text, "html.parser")
		# Get all the text other than html tags.
		stripped_text = soup.get_text(separator=" ")
		return stripped_text
	def remove_links(self, text):
		"""
		This function will remove all the occurrences of links.
		
		arguments:
			input_text: "text" of type "String". 
						
		return:
			value: "text" after removal of all types of links.
			
		Example:
		Input : To know more about this website: kajalyadav.com  visit: https://kajalyadav.com//Blogs
		Output : To know more about this website: visit:     
		
		"""
		
		# Removing all the occurrences of links that starts with https
		remove_https = re.sub(r'http\S+', '', text)
		# Remove all the occurrences of text that ends with .com
		remove_com = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
		return remove_com
	def remove_whitespace(self, text):
		""" This function will remove 
			extra whitespaces from the text
		arguments:
			input_text: "text" of type "String". 
						
		return:
			value: "text" after extra whitespaces removed .
			
		Example:
		Input : How   are   you   doing   ?
		Output : How are you doing ?     
			
		"""
		pattern = re.compile(r'\s+') 
		Without_whitespace = re.sub(pattern, ' ', text)
		# There are some instances where there is no space after '?' & ')', 
		# So I am replacing these with one space so that It will not consider two words as one token.
		text = Without_whitespace.replace('?', ' ? ').replace(')', ') ')
		return text
	# Code for accented characters removal
	def accented_characters_removal(self, text):
		# this is a docstring
		"""
		The function will remove accented characters from the 
		text contained within the Dataset.
		   
		arguments:
			input_text: "text" of type "String". 
						
		return:
			value: "text" with removed accented characters.
			
		Example:
		Input : Málaga, àéêöhello
		Output : Malaga, aeeohello    
			
		"""
		# Remove accented characters from text using unidecode.
		# Unidecode() - It takes unicode data & tries to represent it to ASCII characters. 
		text = unidecode(text)
		return text
	# Code for removing repeated characters and punctuations

	def reducing_incorrect_character_repeatation(self, text):
		"""
		This Function will reduce repeatition to two characters 
		for alphabets and to one character for punctuations.
		
		arguments:
			 input_text: "text" of type "String".
			 
		return:
			value: Finally formatted text with alphabets repeating to 
			two characters & punctuations limited to one repeatition 
			
		Example:
		Input : Realllllllllyyyyy,        Greeeeaaaatttt   !!!!?....;;;;:)
		Output : Reallyy, Greeaatt !?.;:)
		
		"""
		# Pattern matching for all case alphabets
		Pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)
		
		# Limiting all the  repeatation to two characters.
		Formatted_text = Pattern_alpha.sub(r"\1\1", text) 
		
		# Pattern matching for all the punctuations that can occur
		Pattern_Punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')
		
		# Limiting punctuations in previously formatted string to only one.
		Combined_Formatted = Pattern_Punct.sub(r'\1', Formatted_text)
		
		# The below statement is replacing repeatation of spaces that occur more than two times with that of one occurrence.
		Final_Formatted = re.sub(' {2,}',' ', Combined_Formatted)
		return Final_Formatted
	
	# The code for expanding contraction words
	def expand_contractions(self, text):
		"""expand shortened words to the actual form.
		   e.g. don't to do not
		
		   arguments:
				input_text: "text" of type "String".
			 
		   return:
				value: Text with expanded form of shorthened words.
			
		   Example: 
		   Input : ain't, aren't, can't, cause, can't've
		   Output :  is not, are not, cannot, because, cannot have 
		
		 """
		# Tokenizing text into tokens.
		list_Of_tokens = text.split(' ')

		# Checking for whether the given token matches with the Key & replacing word with key's value.
		
		# Check whether Word is in lidt_Of_tokens or not.
		for Word in list_Of_tokens: 
			# Check whether found word is in dictionary "Contraction Map" or not as a key. 
			 if Word in self.CONTRACTION_MAP: 
					# If Word is present in both dictionary & list_Of_tokens, replace that word with the key value.
					list_Of_tokens = [item.replace(Word, self.CONTRACTION_MAP[Word]) for item in list_Of_tokens]
					
		# Converting list of tokens to String.
		String_Of_tokens = ' '.join(str(e) for e in list_Of_tokens) 
		return String_Of_tokens
	# The code for removing special characters
	def removing_special_characters(self, text):
		"""Removing all the special characters except the one that is passed within 
		   the regex to match, as they have imp meaning in the text provided.
	   
		
		arguments:
			 input_text: "text" of type "String".
			 
		return:
			value: Text with removed special characters that don't require.
			
		Example: 
		Input : Hello, K-a-j-a-l. Thi*s is $100.05 : the payment that you will recieve! (Is this okay?) 
		Output :  Hello, Kajal. This is $100.05 : the payment that you will recieve! Is this okay?
		
	   """
		# The formatted text after removing not necessary punctuations.
		Formatted_Text = re.sub(r"[^a-zA-Z0-9:;$-,%.?!]+", ' ', text) 
		# In the above regex expression,I am providing necessary set of punctuations that are frequent in this particular dataset.
		return Formatted_Text
		

	def load_task1(self):
		"""
		Load task 1 training set and convert the tags into binary labels. 
		Paragraphs with original labels of 0 or 1 are considered to be negative examples of PCL and will have the label 0 = negative.
		Paragraphs with original labels of 2, 3 or 4 are considered to be positive examples of PCL and will have the label 1 = positive.
		It returns a pandas dataframe with paragraphs and labels.
		"""
		rows=[]
		with open(os.path.join(self.train_path, 'dontpatronizeme_pcl.tsv')) as f:
			for line in f.readlines()[4:]:
				par_id=line.strip().split('\t')[0]
				art_id = line.strip().split('\t')[1]
				keyword=line.strip().split('\t')[2]
				country=line.strip().split('\t')[3]
				t=line.strip().split('\t')[4].lower()
				l=line.strip().split('\t')[-1]
				if l=='0' or l=='1':
					lbin=0
				else:
					lbin=1
				rows.append(
					{'par_id':par_id,
					'art_id':art_id,
					'keyword':keyword,
					'country':country,
					'text':t, 
					'label':lbin, 
					'orig_label':l
					}
					)
		df=pd.DataFrame(rows, columns=['par_id', 'art_id', 'keyword', 'country', 'text', 'label', 'orig_label']) 
		self.train_task1_df = df

	def load_task2(self, return_one_hot=True):
		# Reads the data for task 2 and present it as paragraphs with binarized labels (a list with seven positions, "activated or not (1 or 0)",
		# depending on wether the category is present in the paragraph).
		# It returns a pandas dataframe with paragraphs and list of binarized labels.
		tag2id = {
				'Unbalanced_power_relations':0,
				'Shallow_solution':1,
				'Presupposition':2,
				'Authority_voice':3,
				'Metaphors':4,
				'Compassion':5,
				'The_poorer_the_merrier':6
				}
		print('Map of label to numerical label:')
		print(tag2id)
		data = defaultdict(list)
		with open (os.path.join(self.train_path, 'dontpatronizeme_categories.tsv')) as f:
			for line in f.readlines()[4:]:
				par_id=line.strip().split('\t')[0]
				art_id = line.strip().split('\t')[1]
				text=line.split('\t')[2].lower()
				keyword=line.split('\t')[3]
				country=line.split('\t')[4]
				start=line.split('\t')[5]
				finish=line.split('\t')[6]
				text_span=line.split('\t')[7]
				label=line.strip().split('\t')[-2]
				num_annotators=line.strip().split('\t')[-1]
				labelid = tag2id[label]
				if not labelid in data[(par_id, art_id, text, keyword, country)]:
					data[(par_id,art_id, text, keyword, country)].append(labelid)

		par_ids=[]
		art_ids=[]
		pars=[]
		keywords=[]
		countries=[]
		labels=[]

		for par_id, art_id, par, kw, co in data.keys():
			par_ids.append(par_id)
			art_ids.append(art_id)
			pars.append(par)
			keywords.append(kw)
			countries.append(co)

		for label in data.values():
			labels.append(label)

		if return_one_hot:
			labels = MultiLabelBinarizer().fit_transform(labels)
		df = pd.DataFrame(list(zip(par_ids, 
									art_ids, 
									pars, 
									keywords,
									countries, 
									labels)), columns=['par_id',
														'art_id', 
														'text', 
														'keyword',
														'country', 
														'label',
														])
		self.train_task2_df = df
		
	def cutBigString(self, aBigString, aContentString, aBefore, aAfter):
		indexBeginSearch = aBigString.find(aContentString)
		indexEndSearch = len(aContentString) + indexBeginSearch
		startSuBstr = indexBeginSearch - aBefore
		if startSuBstr < 0:
			startSuBstr = 0
		endSubStr = indexEndSearch + aAfter;
		maxEndStr = len(aBigString);
		if endSubStr > maxEndStr:
			endSubStr = maxEndStr
		trainText = aBigString[startSuBstr :endSubStr]
		firstIndex = trainText.find(' ')
		lastIndex = trainText.rfind(' ')
		return trainText[firstIndex :lastIndex]	
	
	def load_all_positive(self):
		rows80=[]
		rows20=[]
		index_row = 0
		with open (os.path.join(self.train_path, 'dontpatronizeme_categories.tsv')) as f:
			for line in f.readlines()[4:]:
				index_row = index_row + 1
				par_id=line.strip().split('\t')[0]
				art_id = line.strip().split('\t')[1]
				
				keyword=line.split('\t')[3]
				country=line.split('\t')[4]
				start=line.split('\t')[5]
				finish=line.split('\t')[6]
				text_span=line.split('\t')[7].lower()
				
				text = self.cutBigString(line.split('\t')[2].lower(), text_span, 150, 150)
				
				text = self.expand_contractions(text)
				
				text = "".join([char for char in text if char not in string.punctuation])
				
				text = self.remove_newlines_tabs(text)
				
				text = self.strip_html_tags(text)
				
				text = self.remove_links(text)
				
				text = self.remove_whitespace(text)
				
				text = self.accented_characters_removal(text)
				
				text = self.reducing_incorrect_character_repeatation(text)
				
				text = self.removing_special_characters(text)
				
				label=line.strip().split('\t')[-2]
				num_annotators=line.strip().split('\t')[-1]
				
				if index_row < 2208 :
					rows80.append(
					{'par_id':par_id,
					'art_id':art_id,
					'keyword':keyword,
					'country':country,
					'text':text, 
					'label':label,
					'text_span':text_span
					}
					)
				else :
					rows20.append(
						{'par_id':par_id,
						'art_id':art_id,
						'keyword':keyword,
						'country':country,
						'text':text, 
						'label':label,
						'text_span':text_span
						}
						)
				text = self.cutBigString(line.split('\t')[2].lower(), text_span, 50, 50)
				
				text = self.expand_contractions(text)
				
				text = "".join([char for char in text if char not in string.punctuation])
				
				text = self.remove_newlines_tabs(text)
				
				text = self.strip_html_tags(text)
				
				text = self.remove_links(text)
				
				text = self.remove_whitespace(text)
				
				text = self.accented_characters_removal(text)
				
				text = self.reducing_incorrect_character_repeatation(text)
				
				text = self.removing_special_characters(text)
				if index_row < 2208 :
					rows80.append(
					{'par_id':par_id,
					'art_id':art_id,
					'keyword':keyword,
					'country':country,
					'text':text, 
					'label':label,
					'text_span':text_span
					}
					)
				else :
					rows20.append(
						{'par_id':par_id,
						'art_id':art_id,
						'keyword':keyword,
						'country':country,
						'text':text, 
						'label':label,
						'text_span':text_span
						}
						)
				text = self.cutBigString(line.split('\t')[2].lower(), text_span, 20, 20)
				
				text = self.expand_contractions(text)
				
				text = "".join([char for char in text if char not in string.punctuation])
				
				text = self.remove_newlines_tabs(text)
				
				text = self.strip_html_tags(text)
				
				text = self.remove_links(text)
				
				text = self.remove_whitespace(text)
				
				text = self.accented_characters_removal(text)
				
				text = self.reducing_incorrect_character_repeatation(text)
				
				text = self.removing_special_characters(text)
				if index_row < 2208 :
					rows80.append(
					{'par_id':par_id,
					'art_id':art_id,
					'keyword':keyword,
					'country':country,
					'text':text, 
					'label':label,
					'text_span':text_span
					}
					)
				else :
					rows20.append(
						{'par_id':par_id,
						'art_id':art_id,
						'keyword':keyword,
						'country':country,
						'text':text, 
						'label':label,
						'text_span':text_span
						}
						)
				
		df=pd.DataFrame(rows80, columns=['par_id', 'art_id', 'keyword', 'country', 'text', 'label', 'text_span']) 		
		self._80_positive_df = df
		df=pd.DataFrame(rows20, columns=['par_id', 'art_id', 'keyword', 'country', 'text', 'label', 'text_span']) 		
		self._20_positive_df = df
	def load_all_negative(self):
		rows80=[]
		rows20=[]
		index_row = 0
		with open(os.path.join(self.train_path, 'dontpatronizeme_pcl.tsv')) as f:
			for line in f.readlines()[4:]:
				index_row = index_row + 1
				par_id=line.strip().split('\t')[0]
				art_id = line.strip().split('\t')[1]
				keyword=line.strip().split('\t')[2]
				country=line.strip().split('\t')[3]
				text = line.strip().split('\t')[4].lower()
				text = self.expand_contractions(text)
				
				text = "".join([char for char in text if char not in string.punctuation])
				
				text = self.remove_newlines_tabs(text)
				
				text = self.strip_html_tags(text)
				
				text = self.remove_links(text)
				
				text = self.remove_whitespace(text)
				
				text = self.accented_characters_removal(text)
				
				text = self.reducing_incorrect_character_repeatation(text)
				
				text = self.removing_special_characters(text)
				l=line.strip().split('\t')[-1]
				if l=='0' or l=='1':
					lbin=0
				else:
					continue
				
				if index_row < 7580 :
					rows80.append(
					{'par_id':par_id,
					'art_id':art_id,
					'keyword':keyword,
					'country':country,
					'text':text, 
					'label':lbin, 
					'orig_label':l
					}
					)
				else :				
					rows20.append(
					{'par_id':par_id,
					'art_id':art_id,
					'keyword':keyword,
					'country':country,
					'text':text, 
					'label':lbin, 
					'orig_label':l
					}
					)
		df=pd.DataFrame(rows80, columns=['par_id', 'art_id', 'keyword', 'country', 'text', 'label', 'orig_label']) 
		self._80_negative_df = df
		df=pd.DataFrame(rows20, columns=['par_id', 'art_id', 'keyword', 'country', 'text', 'label', 'orig_label']) 
		self._20_negative_df = df
	
	def load_test(self):
		#self.test_df = [line.strip() for line in open(self.test_path)]
		rows=[]
		with open(self.test_path) as f:
			for line in f.readlines()[4:]:
				t=line.strip().split('\t')[3].lower()
				rows.append(t)
		self.test_set = rows
