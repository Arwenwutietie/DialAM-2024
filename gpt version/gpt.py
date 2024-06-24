import os
import json
import pandas as pd
import re
import openai
from tqdm import tqdm
from openai import AzureOpenAI

folder_path = 'dataset'
df = pd.DataFrame()
pange = pd.DataFrame()

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):  
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as file:
            data = json.load(file)
            df1 = pd.DataFrame.from_dict(data['nodes'])
            pange1 = pd.DataFrame.from_dict(data['edges'])
            # df = df.append(df1)
            # pange = pange.append(pange1)
            df = pd.concat([df, df1])
            pange = pd.concat([pange, pange1])

df = df.drop_duplicates().reset_index(drop=True)
pange = pange.drop_duplicates().reset_index(drop=True)

df['type'].drop_duplicates()
for row in df["text"].head(50):
    print(row)

Snode_table = pd.DataFrame(columns=['from', 'mid', 'last', 'I1', 'I2', 'predict', 'real'])

S_type = ["RA", "MA", "CA"]
mask1 = pange['toID'].isin(pange['fromID'])
mask2 = pange['fromID'].isin(pange[mask1]['toID'])
mask1 &= pange['fromID'].isin(df[df['type'].isin(['I'])]['nodeID'])
mask1 &= pange['toID'].isin(df[df['type'].isin(S_type)]['nodeID'])
mask2 = pange['fromID'].isin(df[df['type'].isin(S_type)]['nodeID'])
mask2 = pange['toID'].isin(df[df['type'].isin(['I'])]['nodeID'])
mask1 &= pange['toID'].isin(pange[mask2]['fromID'])
mask2 &= pange['fromID'].isin(pange[mask1]['toID'])

bef_df = pange[mask1].set_index('toID')
oaft_df = pange[mask2].set_index('fromID')

aft_df = oaft_df.loc[bef_df.index.values].reset_index()
bef_df = bef_df.reset_index()

Snode_table['from'] = bef_df['fromID']
Snode_table['mid'] = bef_df['toID']
Snode_table['last'] = aft_df['toID']
Snode_table['real'] = df.set_index('nodeID').loc[bef_df['toID'].values]['text'].values

Snode_table['I1'] = df.set_index('nodeID').loc[bef_df['fromID'].values]['text'].values
Snode_table['I2'] = df.set_index('nodeID').loc[aft_df['toID'].values]['text'].values

 
Snode_table = Snode_table.reset_index(drop=True)
 
Snode_table["real"].value_counts()

def getNodeSet(t):
    return [t] if t != 'S' else ["RA", "MA", "CA"]

def getChain(f, m, t):
    new_table = pd.DataFrame(columns=['from', 'mid', 'to', f'f{f}', f'm{m}', f't{t}'])
    mask1 = pange['toID'].isin(pange['fromID'])
    mask2 = pange['fromID'].isin(pange[mask1]['toID'])
    mask1 &= pange['fromID'].isin(df[df['type'].isin(getNodeSet(f))]['nodeID'])
    mask1 &= pange['toID'].isin(df[df['type'].isin(getNodeSet(m))]['nodeID'])
    mask2 &= pange['fromID'].isin(df[df['type'].isin(getNodeSet(m))]['nodeID'])
    mask2 &= pange['toID'].isin(df[df['type'].isin(getNodeSet(t))]['nodeID'])
    mask1 &= pange['toID'].isin(pange[mask2]['fromID'])
    mask2 &= pange['fromID'].isin(pange[mask1]['toID'])

    bef_df = pange[mask1].set_index('toID')
    oaft_df = pange[mask2].set_index('fromID')
    oaft_df = oaft_df[~oaft_df.index.duplicated(keep='first')]

    aft_df = oaft_df.loc[bef_df.index.values].reset_index()
    bef_df = bef_df.reset_index()

    new_table['from'] = bef_df['fromID']
    new_table['mid'] = bef_df['toID']
    new_table['to'] = aft_df['toID']

    new_table[f'f{f}'] = df.set_index('nodeID').loc[bef_df['fromID'].values]['text'].values
    new_table[f'm{m}'] = df.set_index('nodeID').loc[bef_df['toID'].values]['text'].values
    new_table[f't{t}'] = df.set_index('nodeID').loc[aft_df['toID'].values]['text'].values

     
    new_table = new_table.reset_index(drop=True)

     
    return new_table

LTL_table = getChain('L', 'TA', 'L')
LYI_table = getChain('L', 'YA', 'I')
ISI_table = getChain('I', 'S', 'I')
TYS_table = getChain('TA', 'YA', 'S')

df = pd.DataFrame({
    'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
    'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
    'rating': [4, 4, 3.5, 15, 5]
})
df = df.set_index('brand')
list(df.index)

LTL_Y_ISI_table = pd.DataFrame(columns=['L1', 'TA', 'L2', 'YA', 'I1', 'S', 'I2'])

TA_indexed_LTL = LTL_table.set_index('mid')
TA_indexed_LTL = TA_indexed_LTL[~TA_indexed_LTL.index.duplicated(keep='first')]
S_index_ISI = ISI_table.set_index('mid')
S_index_ISI = S_index_ISI[~S_index_ISI.index.duplicated(keep='first')]

TYS_table_truncated = TYS_table
TYS_table_truncated = TYS_table_truncated[TYS_table_truncated['from'].isin(TA_indexed_LTL.index)]
TYS_table_truncated = TYS_table_truncated[TYS_table_truncated['to'].isin(S_index_ISI.index)]

LTL_Y_ISI_table['TA'] = TYS_table_truncated['fTA']
LTL_Y_ISI_table['YA'] = TYS_table_truncated['mYA']
LTL_Y_ISI_table['S'] = TYS_table_truncated['tS']

LTL_Y_ISI_table['L1'] = TA_indexed_LTL.loc[TYS_table_truncated['from']]['fL'].values
LTL_Y_ISI_table['L2'] = TA_indexed_LTL.loc[TYS_table_truncated['from']]['tL'].values
LTL_Y_ISI_table['I1'] = S_index_ISI.loc[TYS_table_truncated['to']]['tI'].values
LTL_Y_ISI_table['I2'] = S_index_ISI.loc[TYS_table_truncated['to']]['fI'].values

LTL_Y_ISI_table.to_csv("data_tables/LTL_Y_ISI.csv")

def L_strip(df):
    for index, row in df.iterrows():
        try:
            row['L'] = row['L'].split(':')[1]
        except:
            continue

new_table_na = new_table[1^new_table['real'].isin(['Asserting'])].reset_index(drop=True)
L_strip(new_table_na)

full_table = new_table
L_strip(full_table)

full_table

YA_types = new_table_na.loc[new_table_na['real'].drop_duplicates().index]
for index, row in YA_types.iterrows():
    print(row["real"], f"{row['L']} == {row['I']}")

na_table = new_table_na.sample(frac=1)
full_table = full_table.sample(frac=1)

full_table

prompt_loc = {
        "prototype": r"D:\Academic\HKUST\Courses\Year-2-Spring\UROP 1100\DialAM-2024\Codes\prompt\prototype.txt",
        "zero-shot": r"D:\Academic\HKUST\Courses\Year-2-Spring\UROP 1100\DialAM-2024\Codes\prompt\zero-shot.txt",
        "one-shot": r"D:\Academic\HKUST\Courses\Year-2-Spring\UROP 1100\DialAM-2024\Codes\prompt\one-shot.txt",
        "five-shot": r"D:\Academic\HKUST\Courses\Year-2-Spring\UROP 1100\DialAM-2024\Codes\prompt\five-shot.txt",
        "five-shot-CoT": r"D:\Academic\HKUST\Courses\Year-2-Spring\UROP 1100\DialAM-2024\Codes\prompt\five-shot-CoT.txt",
    }


client = AzureOpenAI(
    api_key="bb2680df57634814964784a1f0836be9",
    api_version="2023-05-15",
    azure_endpoint="https://hkust.azure-api.net"
)

def simple_clean(sentence):
    return re.sub(r'[^\w\s]', '', sentence)
def default_clean(sentence):
    return sentence
def categorical_clean(sentence):
    categories = ["Asserting", "Pure Questioning", "Rhetorical Questioning", "Default Illocuting",
            "Agreeing", "Assertive Questioning", "Arguing", "Restating", "Challenging", "Disagreeing"]
    regex = f"{'|'.join(categories)}"
    return re.search(regex, sentence).group() if re.search(regex, sentence) else 'NA'


class GPT_QUERY:
    def __init__(self, set_size, table):
        self.set_size = set_size
        self.table = table

    def majority_vote(self, votes):
        votes_table = {}
        for vote in votes:
            if vote in votes_table:
                votes_table[vote] += 1
            else:
                votes_table[vote] = 1
        return max(votes_table, key=votes_table.get)

    def gpt(self, x, prompt, processor):
        response = client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[
                {"role": "user", "content": prompt + x + "\nAnswer:"},
            ]
        )
        original = response.choices[0].message.content
        clean_sentence = processor(original)
        return clean_sentence

    def predict(self, prompt_type, processor=categorical_clean, vote=1):
        set_size, table = self.set_size, self.table
        prompt = prompt_loc[prompt_type]
        with tqdm(total=set_size) as bar:
            for index_A, row_A in table.head(set_size).iterrows():
                sentences = [row_A['L'], row_A['I']]
                joined_sentence = " == ".join(sentences)
                try:
                    table.at[index_A, 'predict'] = self.majority_vote([ \
                        self.gpt(joined_sentence, prompt, processor) for _ in range(vote)])
                except Exception as error:
                    print(f"An error occurred at index {index_A}:", error)
                bar.update(1)

        table.head(set_size)[['predict', 'real', 'L', 'I']].to_csv(f"{prompt_type}.csv")
        print(table.head(set_size)[['predict', 'real']])

    def estimate(self, ptf=False):
        set_size, table = self.set_size, self.table
        sum, correct = 0, 0
        for index_A, row_A in table.head(set_size).iterrows():
            sum += 1
            if type(row_A['predict']).__name__ == 'str' and type(row_A['real']).__name__ == 'str' and row_A[
                'predict'].casefold() == row_A['real'].casefold():
                correct = correct + 1
            else:
                if ptf:
                    print("Wrong prediction:", index_A, row_A['predict'], row_A['real'])
        return correct / sum

prompt_list = {name: open(prompt, "r").read() for name, prompt in prompt_loc.items()}
gpt_client = GPT_QUERY(50, full_table)
gpt_client.predict("prototype", vote=3)
gpt_client.estimate()

gpt_client.estimate(True)

# five-shot-CoT-SC

gpt_client = GPT_QUERY(500, full_table)
gpt_client.predict(prompt_list["five-shot-CoT"], vote=3)
gpt_client.estimate()

# five-shot-CoT

gpt_client = GPT_QUERY(250, full_table)
gpt_client.predict(prompt_list["five-shot-CoT"], vote=1)
gpt_client.estimate()

# zero-shot

gpt_client = GPT_QUERY(250, full_table)
gpt_client.predict(prompt_list["zero-shot"])
gpt_client.estimate(ptf=True)

# one-shot

gpt_client = GPT_QUERY(250, full_table)
gpt_client.predict(prompt_list["one-shot"])
gpt_client.estimate(ptf=True)

# five-shot

gpt_client = GPT_QUERY(250, full_table)
gpt_client.predict(prompt_list["five-shot"])
gpt_client.estimate(ptf=True)



