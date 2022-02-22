import os
import string
# Importing Libraries 
import nltk
import re
from unidecode import unidecode
from bs4 import BeautifulSoup 
import sys
#sys.path.append('.')

os.system('python -m pip install contractions')
import contractions

def remove_newlines_tabs(text):
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

def strip_html_tags(text):
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

def remove_links(text):
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

def remove_whitespace(text):
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

def accented_characters_removal(text):
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

def reducing_incorrect_character_repeatation(text):
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
def expand_contractions(text):
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

    return contractions.fix(text)
# The code for removing special characters
def removing_special_characters(text):
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


def apply_preprocessing(text):
    fns = ( remove_newlines_tabs, strip_html_tags, remove_links,
                remove_whitespace, accented_characters_removal,
                reducing_incorrect_character_repeatation, expand_contractions,
                removing_special_characters)
    
    preprocessed = text
    for fn in fns:
        preprocessed = fn(preprocessed)
    return preprocessed

