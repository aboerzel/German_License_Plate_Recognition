from __future__ import division, unicode_literals
import codecs
from bs4 import BeautifulSoup
import json

f = codecs.open("county_license_plates.html", 'r', "utf-8")
soup = BeautifulSoup(f.read(), "lxml")
table = soup.find_all('table')[0]
rows = table.find_all("tr")

data = []

for row in rows:
    cells = row.find_all("td")
    item = {}
    item["LKZ"] = cells[0].text.replace("\r\n", "").replace(" ", "")
    item["DESCR"] = cells[1].text.replace("\r\n", "").replace(" ", "")
    data.append(item)

with open('german_county_marks.json', 'w', encoding='utf8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
