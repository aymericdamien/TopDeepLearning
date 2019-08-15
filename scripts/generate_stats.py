from bs4 import BeautifulSoup
import math
import operator
import requests
import time

def search(keywords, n_pages=10, sort='stars'):
    res = []
    for k in keywords:
        for i in range(1, n_pages+1):
            time.sleep(10)
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
        try:
            stars = str(c.find_all('a', {"class": "muted-link"})[-1].text.strip())
            stars_unparsed = stars
            if 'k' in stars:
                stars = float(stars.replace('k', '')) * 1000
            stars = int(stars)
            lang = c.find('span', {'itemprop': "programmingLanguage"})
            lang = lang.text.strip() if lang else lang
            info.append({
                "url": "https://github.com" + c.find('a', {"class": "v-align-middle"}).attrs["href"],
                "title": c.find('a', {"class": "v-align-middle"}).text.strip(),
                "desc": c.find('p', {"class": "col-12"}).text.strip(),
                "stars_unparsed": stars_unparsed,
                "stars": stars,
                "lang": lang
            })
        except Exception as e:
            print e
    return info

def get_topic(keywords, n_pages=10):
    res = []
    next_token = None
    for k in keywords:
        for i in range(1, n_pages+1):
            time.sleep(10)
            url = "https://github.com/topics/%s" % k
            if next_token:
                url += "?after=%s" % next_token
            r = requests.get(url)
            info, next_token = extract_topic_info(r.content)
            for r in info:
                res.append(r)
    return res

def extract_topic_info(html):
    info = []
    html = BeautifulSoup(html, 'html.parser')
    next_token = html.find('input', {"name": "after"}).attrs["value"]
    for c in html.find_all('article', {"class": "py-4"}):
        try:
            stars = str(c.find_all('a', {"class": "link-gray"})[-1].text.strip())
            stars_unparsed = stars
            if 'k' in stars:
                stars = float(stars.replace('k', '')) * 1000
            stars = int(stars)
            lang = c.find('span', {'itemprop': "programmingLanguage"})
            lang = lang.text.strip() if lang else lang
            info.append({
                "url": "https://github.com" + c.find('h3').find('a').attrs["href"],
                "title": c.find('h3').find('a').text.strip().replace(" / ", "/"),
                "desc": c.find('div', {"class": "mb-3"}).text.strip(),
                "stars_unparsed": stars_unparsed,
                "stars": stars,
                "lang": lang
            })
        except Exception as e:
            print e
    return info, next_token

def parse_results(results):
    results = {v['url']:v for v in results}.values()
    results = sorted(results, key=lambda x: x['stars'], reverse=True)
    return [r for r in results if r['stars'] >= 1000 and (r['lang'] and r['lang'] != "TeX")]

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

topics = get_topic(['tensorflow', 'deep-learning', 'pytorch', 'machine-learning'], n_pages=5)
searches = search(['tensorflow', 'deep learning', 'pytorch', 'cntk', 'machine learning'], n_pages=5)

r = parse_results(topics + searches)

print len(r)

with open('out.html', 'w') as f:
    f.write(build_table(r)[0].encode('utf-8'))

with open('out.md', 'w') as f:
    f.write(build_table(r)[1].encode('utf-8'))
