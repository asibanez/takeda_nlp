#Imports
import nltk
import codecs
import pandas as pd

# Path definitions
input_path = 'C:\\Users\\siban\\Desktop\\00_takeda\\00_data\\01_results.txt'
output_path = 'C:\\Users\\siban\\Desktop\\00_takeda\\00_data\\preprocessed.txt'

# Data loading
data = pd.read_csv(input_path, keep_default_na = False)

#%%
# Variable initizalization
merge_data = True    # Combines annotations in the same paragraph
clean_par = True     # Removes title annonations
tgt_vars = ['title', 'authors', 'journal', 'study_type', 'arm_description',
            'arm_dosage', 'arm_efficacy_metric', 'arm_efficacy_results']

#%% Clean title entries to remove duplicated annotations
if clean_par == True:
    for i, row in data.iterrows():
        if row['title'] != '':
            data.at[i, 'title'] = ''
            data.at[i, 'title-tag'] = ''

#%% Merge paragraphs for NER

if merge_data == True:

    merged_data = pd.DataFrame(columns = data.columns)
    merged_data = pd.concat([merged_data, data[0:1]])
    
    for idx in range(1, data.shape[0]):
        pos = merged_data.shape[0] - 1
        condition_1 = data.iloc[idx]['doc_id'] == data.iloc[idx - 1]['doc_id']
        condition_2 = data.iloc[idx]['description'] == data.iloc[idx - 1]['description']
        
        if condition_1 and condition_2:
            
            merged_data.at[pos, 'title-tag'] = (merged_data.iloc[pos]['title-tag'] + ',' + data.iloc[idx]['title-tag']).strip(',')
            merged_data.at[pos, 'authors-tag'] = (merged_data.iloc[pos]['authors-tag'] + ',' + data.iloc[idx]['authors-tag']).strip(',')
            merged_data.at[pos, 'journal-tag'] = (merged_data.iloc[pos]['journal-tag'] + ',' + data.iloc[idx]['journal-tag']).strip(',')
            merged_data.at[pos, 'study_type-tag'] = (merged_data.iloc[pos]['study_type-tag'] + ',' + data.iloc[idx]['study_type-tag']).strip(',')
            merged_data.at[pos, 'arm_description-tag'] = (merged_data.iloc[pos]['arm_description-tag'] + ',' + data.iloc[idx]['arm_description-tag']).strip(',')
            merged_data.at[pos, 'arm_dosage-tag'] = (merged_data.iloc[pos]['arm_dosage-tag'] + ',' + data.iloc[idx]['arm_dosage-tag']).strip(',')
            merged_data.at[pos, 'arm_efficacy_metric-tag'] = (merged_data.iloc[pos]['arm_efficacy_metric-tag'] + ',' + data.iloc[idx]['arm_efficacy_metric-tag']).strip(',')
            merged_data.at[pos, 'arm_efficacy_results-tag'] = (merged_data.iloc[pos]['arm_efficacy_results-tag'] + ',' + data.iloc[idx]['arm_efficacy_results-tag']).strip(',')
            
            merged_data.at[pos, 'title'] = (merged_data.iloc[pos]['title'] + ' ' + data.iloc[idx]['title']).strip()
            merged_data.at[pos, 'authors'] = (merged_data.iloc[pos]['authors'] + ' ' + data.iloc[idx]['authors']).strip(',')
            merged_data.at[pos, 'journal'] = (merged_data.iloc[pos]['journal'] + ' ' + data.iloc[idx]['journal']).strip(',')
            merged_data.at[pos, 'study_type'] = (merged_data.iloc[pos]['study_type'] + ' ' + data.iloc[idx]['study_type']).strip(',')
            merged_data.at[pos, 'arm_description'] = (merged_data.iloc[pos]['arm_description'] + ' ' + data.iloc[idx]['arm_description']).strip(',')
            merged_data.at[pos, 'arm_dosage'] = (merged_data.iloc[pos]['arm_dosage'] + ' ' + data.iloc[idx]['arm_dosage']).strip(',')
            merged_data.at[pos, 'arm_efficacy_metric'] = (merged_data.iloc[pos]['arm_efficacy_metric'] + ' ' + data.iloc[idx]['arm_efficacy_metric']).strip(',')
            merged_data.at[pos, 'arm_efficacy_results'] = (merged_data.iloc[pos]['arm_efficacy_results'] + ' ' + data.iloc[idx]['arm_efficacy_results']).strip(',')
    
        else:
            merged_data = pd.concat([merged_data, data[idx:idx + 1]]).reset_index(drop = True)

    data_old = data    
    data = merged_data
        
#%% Open output file
fw = codecs.open(output_path, 'w', 'utf-8')

# Iterate over rows in dataframe
for idx, row in data.iterrows():
    par = row['description']
    #par_tokens = nltk.word_tokenize(par)
    par_tokens = par.split(' ')
    
    # Initialize token tags to 'O'
    par_tags = ['O'] * len(par_tokens)
    
    # Iterate over tgt variables
    for tgt_var in tgt_vars:
        annot_tgt_var = row[tgt_var]  
        annot_spans_list = row[tgt_var + '-tag'].split(',')
        
        # Skip empty fields
        if annot_tgt_var != '':
            #annot_tokens = nltk.word_tokenize(annot_tgt_var)
            annot_tokens = annot_tgt_var.split(' ')
            annot_spans = [(int(x), int(y)) for x, y in zip(annot_spans_list[::2],
                                                            annot_spans_list[1::2])]
            # Iterate over spans in field
            for span in annot_spans:
                start, end = span
                par_tags[start] = 'B-' + tgt_var
                if end == start + 1:
                    continue
                for i in range(start+1, end):
                    par_tags[i] = 'I-' + tgt_var
        else:
            continue
            
    # Write output file
    for token, tag in zip(par_tokens, par_tags):
        fw.write(str(idx) + '\t' + token + '\t' + tag + '\n')
    fw.write('--------------------------------------------------------------\n')

# Close output file
fw.close()

