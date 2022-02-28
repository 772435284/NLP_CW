import os
import random
import subprocess
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=os.path.join(os.path.dirname(__file__), "tangzifeng1998827-12d0a6759fc6.json")
# subprocess.call("python -m pip install google-cloud-translate==2.0.1".split())
import six

import six
from google.cloud import translate_v2 as translate

# os.system("python -m pip install google-cloud-translate==2.0.1")

def translate_text(text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """


    translate_client = translate.Client()
    targets = ['zh-CN', 'af', 'de', 'fr', 'nl', 'pt', 'pl', 'es', 'ru', 'el']
    target = targets[random.randint(0, len(targets)-1)]
    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")
    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    back_translated_result = translate_client.translate(result["translatedText"], target_language='en')
    # print(u"Text: {}".format(result["input"]))
    # print(u"Translation: {}".format(result["translatedText"]))
    # print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))
    # print(u"Target: {}".format(target))
    # print(u"Back translated language: {}".format(back_translated_result["translatedText"]))
    return back_translated_result['translatedText']

print(translate_text("fuck you!"))