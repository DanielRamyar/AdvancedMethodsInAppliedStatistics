import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')

my_url = 'https://kenpom.com/index.php?y=2014'
my_page = requests.get(my_url)

soup = BeautifulSoup(my_page.text, 'html.parser')

ratings_table = soup.find(id='ratings-table')

headers = ratings_table.find(class_='thead2')

titles = headers.find_all('th')

header_titles = []

for title in titles:
    try:
        header_titles.append(title.contents[0].contents[0])
    except:
        header_titles.append(title.contents[0])

print(header_titles)

body = ratings_table.find_all('tbody')
rows = body[0].find_all('tr')

conferences = ['ACC', 'SEC', 'B10', 'BSky', 'A10', 'BE']
ACC_AdjD = np.array([])
SEC_AdjD = np.array([])
B10_AdjD = np.array([])
BSky_AdjD = np.array([])
A10_AdjD = np.array([])
BE_AdjD = np.array([])

for row in rows:
    temp = row.find_all('td')
    if len(temp) == 0:
        pass
    else:
        if temp[2].contents[0].contents[0] in conferences:
            if str(temp[2].contents[0].contents[0]) == 'ACC':
                ACC_AdjD = np.append(ACC_AdjD, float(temp[7].contents[0]))
            elif str(temp[2].contents[0].contents[0]) == 'SEC':
                SEC_AdjD = np.append(SEC_AdjD, float(temp[7].contents[0]))
            elif str(temp[2].contents[0].contents[0]) == 'B10':
                B10_AdjD = np.append(B10_AdjD, float(temp[7].contents[0]))
            elif str(temp[2].contents[0].contents[0]) == 'BSky':
                BSky_AdjD = np.append(BSky_AdjD, float(temp[7].contents[0]))
            elif str(temp[2].contents[0].contents[0]) == 'A10':
                A10_AdjD = np.append(A10_AdjD, float(temp[7].contents[0]))
            elif str(temp[2].contents[0].contents[0]) == 'BE':
                BE_AdjD = np.append(BE_AdjD, float(temp[7].contents[0]))
        else:
            pass


binwidth = 0.5
n_bins_1 = np.arange(min(ACC_AdjD), max(ACC_AdjD) + binwidth, binwidth)
n_bins_2 = np.arange(min(SEC_AdjD), max(SEC_AdjD) + binwidth, binwidth)
n_bins_3 = np.arange(min(B10_AdjD), max(B10_AdjD) + binwidth, binwidth)
n_bins_4 = np.arange(min(BSky_AdjD), max(BSky_AdjD) + binwidth, binwidth)
n_bins_5 = np.arange(min(A10_AdjD), max(A10_AdjD) + binwidth, binwidth)
n_bins_6 = np.arange(min(BE_AdjD), max(BE_AdjD) + binwidth, binwidth)


plt.hist(ACC_AdjD, bins=n_bins_1, alpha=0.9, color='blue', edgecolor='black', linewidth=1, label='ACC')
plt.hist(SEC_AdjD, bins=n_bins_2, color='maroon', edgecolor='black', linewidth=1, label='SEC')
plt.hist(B10_AdjD, bins=n_bins_3, alpha=0.9, color='yellow', edgecolor='black', linewidth=1, label='B10')
plt.hist(BSky_AdjD, bins=n_bins_4, color='green', edgecolor='black', linewidth=1, label='BSky')
plt.hist(A10_AdjD, bins=n_bins_5, alpha=0.7, color='grey', edgecolor='black', linewidth=1, label='A10')
plt.hist(BE_AdjD, bins=n_bins_5, alpha=0.5, color='m', edgecolor='black', linewidth=1, label='BE')

plt.title("The Adjusted Defense for 6 conferences")
plt.legend()
plt.ylabel('Number of counts in bin')
plt.xlabel('AdjD')

plt.show()
