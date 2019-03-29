import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')

### Getting 2014 data
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

conferences = ['ACC', 'SEC', 'B10', 'BSky', 'A10']
ACC_teams_2014 = np.array([])
SEC_teams_2014 = np.array([])
B10_teams_2014 = np.array([])
BSky_teams_2014 = np.array([])
A10_teams_2014 = np.array([])
rest_teams_2014 = np.array([])

ACC_AdjO_2014 = np.array([])
SEC_AdjO_2014 = np.array([])
B10_AdjO_2014 = np.array([])
BSky_AdjO_2014 = np.array([])
A10_AdjO_2014 = np.array([])
rest_AdjO_2014 = np.array([])


for row in rows:
    temp = row.find_all('td')
    if len(temp) == 0:
        pass
    else:
        if temp[2].contents[0].contents[0] in conferences:
            if str(temp[2].contents[0].contents[0]) == 'ACC':
                ACC_teams_2014 = np.append(ACC_teams_2014, temp[1].contents[0].contents[0])
                ACC_AdjO_2014 = np.append(ACC_AdjO_2014, float(temp[5].contents[0]))
            elif str(temp[2].contents[0].contents[0]) == 'SEC':
                SEC_teams_2014 = np.append(SEC_teams_2014, temp[1].contents[0].contents[0])
                SEC_AdjO_2014 = np.append(SEC_AdjO_2014, float(temp[5].contents[0]))
            elif str(temp[2].contents[0].contents[0]) == 'B10':
                B10_teams_2014 = np.append(B10_teams_2014, temp[1].contents[0].contents[0])
                B10_AdjO_2014 = np.append(B10_AdjO_2014, float(temp[5].contents[0]))
            elif str(temp[2].contents[0].contents[0]) == 'BSky':
                BSky_teams_2014 = np.append(BSky_teams_2014, temp[1].contents[0].contents[0])
                BSky_AdjO_2014 = np.append(BSky_AdjO_2014, float(temp[5].contents[0]))
            elif str(temp[2].contents[0].contents[0]) == 'A10':
                A10_teams_2014 = np.append(A10_teams_2014, temp[1].contents[0].contents[0])
                A10_AdjO_2014 = np.append(A10_AdjO_2014, float(temp[5].contents[0]))
        else:
            rest_teams_2014 = np.append(rest_teams_2014, temp[1].contents[0].contents[0])
            rest_AdjO_2014 = np.append(rest_AdjO_2014, float(temp[5].contents[0]))

###### Getting 2009 data
my_url = 'https://kenpom.com/index.php?y=2009'
my_page = requests.get(my_url)

soup = BeautifulSoup(my_page.text, 'html.parser')
ratings_table = soup.find(id='ratings-table')

body = ratings_table.find_all('tbody')
rows = body[0].find_all('tr')

ACC_teams_2009 = np.array([])
SEC_teams_2009 = np.array([])
B10_teams_2009 = np.array([])
BSky_teams_2009 = np.array([])
A10_teams_2009 = np.array([])
rest_teams_2009 = np.array([])

ACC_AdjO_2009 = np.array([])
SEC_AdjO_2009 = np.array([])
B10_AdjO_2009 = np.array([])
BSky_AdjO_2009 = np.array([])
A10_AdjO_2009 = np.array([])
rest_AdjO_2009 = np.array([])

for row in rows:
    temp = row.find_all('td')
    if len(temp) == 0:
        pass
    else:
        if temp[2].contents[0].contents[0] in conferences:
            if str(temp[2].contents[0].contents[0]) == 'ACC':
                ACC_teams_2009 = np.append(ACC_teams_2009, temp[1].contents[0].contents[0])
                ACC_AdjO_2009 = np.append(ACC_AdjO_2009, float(temp[5].contents[0]))
            elif str(temp[2].contents[0].contents[0]) == 'SEC':
                SEC_teams_2009 = np.append(SEC_teams_2009, temp[1].contents[0].contents[0])
                SEC_AdjO_2009 = np.append(SEC_AdjO_2009, float(temp[5].contents[0]))
            elif str(temp[2].contents[0].contents[0]) == 'B10':
                B10_teams_2009 = np.append(B10_teams_2009, temp[1].contents[0].contents[0])
                B10_AdjO_2009 = np.append(B10_AdjO_2009, float(temp[5].contents[0]))
            elif str(temp[2].contents[0].contents[0]) == 'BSky':
                BSky_teams_2009 = np.append(BSky_teams_2009, temp[1].contents[0].contents[0])
                BSky_AdjO_2009 = np.append(BSky_AdjO_2009, float(temp[5].contents[0]))
            elif str(temp[2].contents[0].contents[0]) == 'A10':
                A10_teams_2009 = np.append(A10_teams_2009, temp[1].contents[0].contents[0])
                A10_AdjO_2009 = np.append(A10_AdjO_2009, float(temp[5].contents[0]))
        else:
            rest_teams_2009 = np.append(rest_teams_2009, temp[1].contents[0].contents[0])
            rest_AdjO_2009 = np.append(rest_AdjO_2009, float(temp[5].contents[0]))

diff_ACC_teams = np.array([])
diff_ACC_AdjO = np.array([])
ACC_Adjo_2009_value = np.array([])

for i, team_2014 in enumerate(ACC_teams_2014, 0):
    for j, team_2009 in enumerate(ACC_teams_2009, 0):
        if team_2014 == team_2009:
            diff_ACC_teams = np.append(diff_ACC_teams, team_2014)
            ACC_Adjo_2009_value = np.append(ACC_Adjo_2009_value, ACC_AdjO_2009[j])
            diff_ACC_AdjO = np.append(diff_ACC_AdjO, ACC_AdjO_2014[i] - ACC_AdjO_2009[j])

diff_SEC_teams = np.array([])
diff_SEC_AdjO = np.array([])
SEC_Adjo_2009_value = np.array([])

for i, team_2014 in enumerate(SEC_teams_2014, 0):
    for j, team_2009 in enumerate(SEC_teams_2009, 0):
        if team_2014 == team_2009:
            diff_SEC_teams = np.append(diff_SEC_teams, team_2014)
            SEC_Adjo_2009_value = np.append(SEC_Adjo_2009_value, SEC_AdjO_2009[j])
            diff_SEC_AdjO = np.append(diff_SEC_AdjO, SEC_AdjO_2014[i] - SEC_AdjO_2009[j])

diff_B10_teams = np.array([])
diff_B10_AdjO = np.array([])
B10_Adjo_2009_value = np.array([])

for i, team_2014 in enumerate(B10_teams_2014, 0):
    for j, team_2009 in enumerate(B10_teams_2009, 0):
        if team_2014 == team_2009:
            diff_B10_teams = np.append(diff_B10_teams, team_2014)
            B10_Adjo_2009_value = np.append(B10_Adjo_2009_value, B10_AdjO_2009[j])
            diff_B10_AdjO = np.append(diff_B10_AdjO, B10_AdjO_2014[i] - B10_AdjO_2009[j])

diff_BSky_teams = np.array([])
diff_BSky_AdjO = np.array([])
BSky_Adjo_2009_value = np.array([])

for i, team_2014 in enumerate(BSky_teams_2014, 0):
    for j, team_2009 in enumerate(BSky_teams_2009, 0):
        if team_2014 == team_2009:
            diff_BSky_teams = np.append(diff_BSky_teams, team_2014)
            BSky_Adjo_2009_value = np.append(BSky_Adjo_2009_value, BSky_AdjO_2009[j])
            diff_BSky_AdjO = np.append(diff_BSky_AdjO, BSky_AdjO_2014[i] - BSky_AdjO_2009[j])

diff_A10_teams = np.array([])
diff_A10_AdjO = np.array([])
A10_Adjo_2009_value = np.array([])

for i, team_2014 in enumerate(A10_teams_2014, 0):
    for j, team_2009 in enumerate(A10_teams_2009, 0):
        if team_2014 == team_2009:
            diff_A10_teams = np.append(diff_A10_teams, team_2014)
            A10_Adjo_2009_value = np.append(A10_Adjo_2009_value, A10_AdjO_2009[j])
            diff_A10_AdjO = np.append(diff_A10_AdjO, A10_AdjO_2014[i] - A10_AdjO_2009[j])

diff_rest_teams = np.array([])
diff_rest_AdjO = np.array([])
rest_Adjo_2009_value = np.array([])

for i, team_2014 in enumerate(rest_teams_2014, 0):
    for j, team_2009 in enumerate(rest_teams_2009, 0):
        if team_2014 == team_2009:
            diff_rest_teams = np.append(diff_rest_teams, team_2014)
            rest_Adjo_2009_value = np.append(rest_Adjo_2009_value, rest_AdjO_2009[j])
            diff_rest_AdjO = np.append(diff_rest_AdjO, rest_AdjO_2014[i] - rest_AdjO_2009[j])

plt.plot(ACC_Adjo_2009_value, diff_ACC_AdjO, alpha=0.9, color='blue', marker='o', linestyle='none', label='ACC')
plt.plot(SEC_Adjo_2009_value, diff_SEC_AdjO, color='maroon', marker='o', linestyle='none', label='SEC')
plt.plot(B10_Adjo_2009_value, diff_B10_AdjO, alpha=1, color='yellow', markeredgecolor='black', marker='o', linestyle='none', label='B10')
plt.plot(BSky_Adjo_2009_value, diff_BSky_AdjO, color='green', marker='o', linestyle='none', label='BSky')
plt.plot(A10_Adjo_2009_value, diff_A10_AdjO, alpha=1, color='grey', markeredgecolor='black', marker='o', linestyle='none', label='A10')

plt.title("AdjO difference for teams in 5 conferences in both 2009 and 2014 datasets")



plt.ylabel('AdjO difference (2014-2009)')
plt.xlabel('2009 AdjO value')

plt.legend()
plt.show()

ACC_mean = np.mean(diff_ACC_AdjO)
ACC_median = np.median(diff_ACC_AdjO)
print("Mean for ACC is: %8.2f" % (ACC_mean))
print("Median for ACC is: %8.2f" % (ACC_median))

SEC_mean = np.mean(diff_SEC_AdjO)
SEC_median = np.median(diff_SEC_AdjO)
print("Mean for SEC is: %8.2f" % (SEC_mean))
print("Median for SEC is: %8.2f" % (SEC_median))

B10_mean = np.mean(diff_B10_AdjO)
B10_median = np.median(diff_B10_AdjO)
print("Mean for B10 is: %8.2f" % (B10_mean))
print("Median for B10 is: %8.2f" % (B10_median))

BSky_mean = np.mean(diff_BSky_AdjO)
BSky_median = np.median(diff_BSky_AdjO)
print("Mean for BSky is: %8.2f" % (BSky_mean))
print("Median for BSky is: %8.2f" % (BSky_median))

A10_mean = np.mean(diff_A10_AdjO)
A10_median = np.median(diff_A10_AdjO)
print("Mean for A10 is: %8.2f" % (A10_mean))
print("Median for A10 is: %8.2f" % (A10_median))

rest_mean = np.mean(diff_rest_AdjO)
rest_median = np.median(diff_rest_AdjO)
print("Mean for rest is: %8.2f" % (rest_mean))
print("Median for rest is: %8.2f" % (rest_median))
