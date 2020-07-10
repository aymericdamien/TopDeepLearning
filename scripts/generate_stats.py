from bs4 import BeautifulSoup
import math
import operator
import requests
import time

SKIP_LIST = ["awesome", "notebook", "learn", "curated list"]

def search(keywords, n_pages=10, sort='stars'):
    res = []
    for k in keywords:
        for i in range(1, n_pages+1):
            time.sleep(20)
            url = "https://github.com/search?o=desc&q=%s&p=%d&s=%s&type=Repositories" % (k, i, sort)
            r = requests.get(url)
            info = extract_search_info(r.content)
            for r in info:
                res.append(r)
    return res

def extract_search_info(html):
    info = []
    html = BeautifulSoup(html, 'html.parser')
    for c in html.find_all('li', {"class": "repo-list-item"}):
        url = None
        try:
            stars = str(c.find_all('a', {"class": "muted-link"})[-1].text.strip())
            stars_unparsed = stars
            if 'k' in stars:
                stars = float(stars.replace('k', '')) * 1000
            stars = int(stars)
            lang = None
            url = "https://github.com" + c.find('a', {"class": "v-align-middle"}).attrs["href"]
            title = c.find('a', {"class": "v-align-middle"}).text.strip().split('/')[-1]
            desc = c.find('p', {"class": "mb-1"}).text.strip()
            skip = 0
            for w in SKIP_LIST:
                if w in desc:
                    skip = 1
                    break
            if skip == 0:
              info.append({
                "url": url,
                "title": title,
                "desc": desc.strip(),
                "stars_unparsed": stars_unparsed,
                "stars": stars,
                "lang": lang
              })
        except Exception as e:
            print url
            print e
    return info

def get_topic(keywords, n_pages=10):
    res = []
    next_token = None
    for k in keywords:
        for i in range(1, n_pages+1):
            time.sleep(20)
            url = "https://github.com/topics/%s?page=%i" % (k, i)
            r = requests.get(url)
            info = extract_topic_info(r.content)
            for r in info:
                res.append(r)
    return res

def extract_topic_info(html):
    info = []
    html = BeautifulSoup(html, 'html.parser')
    for c in html.find_all('article', {"class": "my-4"}):
        url = None
        try:
            stars = str(c.find_all('a', {"class": "social-count"})[-1].text.strip())
            stars_unparsed = stars
            if 'k' in stars:
                stars = float(stars.replace('k', '')) * 1000
            stars = int(stars)
            lang = None
            url = "https://github.com" + c.find('h1').find_all('a')[1].attrs["href"]
            time.sleep(5)
            r = requests.get(url)
            desc = BeautifulSoup(r.content, 'html.parser').find('p', {'class': 'f4'}).text
            skip = 0
            for w in SKIP_LIST:
                if w in desc:
                    skip = 1
                    break
            if skip == 0:
              info.append({
                "url": url,
                "title": c.find('h1').find_all('a')[1].text.strip().replace(" / ", "/"),
                "desc": desc.strip(),
                "stars_unparsed": stars_unparsed,
                "stars": stars,
                "lang": lang
              })
        except Exception as e:
            print url
            print e
    return info

def parse_results(results):
    results = {v['url']:v for v in results}.values()
    results = sorted(results, key=lambda x: x['stars'], reverse=True)
    return [r for r in results if r['stars'] >= 1000]

def build_table(results_list):

    def build_html_fields(d):
        return ['<a href="%s">%s</a>' % (d['url'], d['title'].split('/')[-1]), d['stars_unparsed'], d['desc']]

    def build_md_fields(d):
        return ['[%s](%s)' % (d['title'].split('/')[-1], d['url']), d['stars_unparsed'], d['desc']]

    html = '<table><thead><tr><td>Project Name</td><td>Stars</td><td>Description</td></tr></thead>'
    md = '| Project Name | Stars | Description |\n| ------- | ------ | ------ |\n'
    for r in results_list:
        html += '<tr><td>' + '</td><td>'.join(build_html_fields(r)) + '</td></tr>'
        md += '|' + '|'.join(build_md_fields(r)) + '|\n'
    html += '</table>'
    return html, md

topics = get_topic(['tensorflow', 'deep-learning', 'pytorch', 'machine-learning'], n_pages=15)
searches = search(['tensorflow', 'deep learning', 'pytorch', 'cntk', 'machine learning'], n_pages=15)

r = parse_results(topics + searches)

print len(r)

with open('out.html', 'w') as f:
    f.write(build_table(r)[0].encode('utf-8'))

with open('out.md', 'w') as f:
    f.write(build_table(r)[1].encode('utf-8'))
