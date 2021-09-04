import pandas as pd
import re
import textdistance
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process 
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
amazon_data=pd.read_csv('amazon.csv',encoding = 'ISO-8859-1')
google_data=pd.read_csv('google.csv',encoding = 'ISO-8859-1')

#textpreprocessing
unwanted_pattern = r'[\/\-()\.,:+0-9;?&!\']' #matches /,-,(), .words
amazon_processed = {}
google_processed = {}

output_amazon = []
output_google = []

count = 0
while count < len(amazon_data['idAmazon']):
    new_string = re.sub(unwanted_pattern, '', str(amazon_data['title'][count]))
    amazon_processed[amazon_data['idAmazon'][count]] = ' '.join([word for word in new_string.split() if word not in stopwords])
    count = count + 1
    
count = 0
while count < len(google_data['id']):
    new_string = re.sub(unwanted_pattern, '', str(google_data['name'][count]))
    google_processed[google_data['id'][count]] = ' '.join([word for word in new_string.split() if word not in stopwords])
    count = count + 1

for key,sentence in amazon_processed.items():
    for word in sentence.split():
        output_amazon.append([word,key])
        #if key not in output_amazon:
            #output_amazon[key] = [word]
        #elif key in output_amazon:
            #output_amazon[key].append(word)

for key,sentence in google_processed.items():
    for word in sentence.split():
        output_google.append([word,key])
        #if key not in output_google:
                #output_google[key] = [word]
        #elif key in output_google:
            #output_google[key].append(word)

#print(output_amazon)
#print('GOOGLE', output_google)
#Exports task1b
amazon_df = pd.DataFrame(output_amazon, columns = ['block_key', 'product_id']) 
amazon_df.to_csv('amazon_blocks.csv', index=False)

google_df = pd.DataFrame(output_google, columns = ['block_key', 'product_id']) 
google_df.to_csv('google_blocks.csv', index=False)

#google_dict = {'block_key': list(output_google.keys()), 'product_id': list(output_google.values())}
#google_df = pd.DataFrame(google_dict)
#google_df.to_csv('google_blocks.csv', index=False)