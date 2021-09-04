#Function to calculate Jaccard Similarity
def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

import pandas as pd
import re
import textdistance
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

#Reads csv
amazonsmall_data=pd.read_csv('amazon_small.csv',encoding = 'ISO-8859-1')
googlesmall_data=pd.read_csv('google_small.csv',encoding = 'ISO-8859-1')

amazon_title = amazonsmall_data['title']
amazon_preprocessed = {}
google_title = googlesmall_data['name']
google_preprocessed = {}

#Text preprocessing for both amazon and google csv
unwanted_pattern = r'[\/\-()\.,:+]' #matches /,-,(), .

count = 0
while count < len(amazonsmall_data['idAmazon']):
    new_string = re.sub(unwanted_pattern, '', str(amazon_title[count]))
    amazon_preprocessed[amazonsmall_data['idAmazon'][count]] = ' '.join([word for word in new_string.split() if word not in stopwords])
    count = count + 1
    
count = 0
while count < len(googlesmall_data['idGoogleBase']):
    new_string = re.sub(unwanted_pattern, '', str(google_title[count]))
    google_preprocessed[googlesmall_data['idGoogleBase'][count]] = ' '.join([word for word in new_string.split() if word not in stopwords])
    count = count + 1

idgoogle_similar = []
idamazon_similar = []
#Compare similarities between each rows in both csv
for (id_amazon,title) in amazon_preprocessed.items():
    maximum = 0
    maxvalue_dictionary = {}
    for (id_google,name) in google_preprocessed.items():
        jaccard_score = get_jaccard_sim(title, name)
        #Finds all the Jaccard Score and idGoogle that are higher than the previous score
        if jaccard_score > maximum:
            maximum = jaccard_score
            maxvalue_dictionary[jaccard_score] = id_google
    if maxvalue_dictionary:
        all_scores =  maxvalue_dictionary.keys()
        if  max(all_scores)>= 0.22: #Concludes a match if maximum score in temporary dictionary exceeds threshold 0.22
            idamazon_similar.append(id_amazon)
            all_scores =  maxvalue_dictionary.keys()
            idgoogle_similar.append(maxvalue_dictionary[max(all_scores)])

#Export results to csv
task1a_dict = {'idAmazon': idamazon_similar, 'idGoogleBase': idgoogle_similar}
task1a_df = pd.DataFrame(task1a_dict)
task1a_df.to_csv('task1a.csv', index=False)
