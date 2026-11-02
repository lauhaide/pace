import jsonlines
import sys
from tqdm import tqdm

from .abstract_generator import AbstractGenerator
from typing import Dict, List

#https://trafilatura.readthedocs.io/en/latest/installation.html
#pip install trafilatura
#pip install --upgrade trafilatura
#pip install --force-reinstall -U git+https://github.com/adbar/trafilatura
from trafilatura import extract
import re


# Patterns
SCRIPT_PATTERN = r"<[ ]*script.*?\/[ ]*script[ ]*>"
STYLE_PATTERN = r"<[ ]*style.*?\/[ ]*style[ ]*>"
META_PATTERN = r"<[ ]*meta.*?>"
COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
LINK_PATTERN = r"<[ ]*link.*?>"
BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
SVG_PATTERN = r"(<svg[^>]*>)(.*?)(<\/svg>)"

SKIP_URLS= ["https://www.mirror.co.uk/3am/celebrity-news/dwayne-johnsons-screwed-up-first-24783522", \
            "https://www.footballcritic.com/stade-brestois-29/squad/587"]

def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
    return re.sub(
        SVG_PATTERN,
        lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
        html,
        flags=re.DOTALL,
    )


def replace_base64_images(html: str, new_image_src: str = "#") -> str:
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)


def clean_html(html: str, clean_svg: bool = False, clean_base64: bool = False):
    html = re.sub(
        SCRIPT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        STYLE_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        META_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        COMMENT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        LINK_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    if clean_svg:
        html = replace_svg(html)
    if clean_base64:
        html = replace_base64_images(html)
    return html

class CRAG_Generator(AbstractGenerator):

    def get_page_text(self, html):
        html_encoding = EncodingDetector.find_declared_encoding(html, is_html=True)
        soup = BeautifulSoup(html, features="html.parser", from_encoding=html_encoding)
        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # rip it out
        # are there tables
        all_tabs = []
        for tab in soup(["table"]):
            print('TABLE...')
            parent = tab.parent
            tab = tab.extract()    # rip it out
            caption = tab.find('caption')
            tab_lines = [parent.get_text().strip()]
            if caption:
                tab_lines.append(caption.get_text().strip())
            parent.extract()
            for tag in tab.descendants:
                try:
                    tag.attrs.clear()
                except:
                    pass
            for a in tab.findAll('a'):
                a.extract()
            for tr in tab.find_all('tr'):
                head = [th.text.strip() for th in tr.find_all('th')] 
                tab_lines.append(" | ".join(head))
                row = [td.text.strip() for td in tr.find_all('td')]
                tab_lines.append(" | ".join(row))
            #print("\n".join(tab_lines))
            #break
            all_tabs.append("\n".join(tab_lines))

        for tag in soup.findAll():
            try:
                tag.attrs.clear()
            except:
                pass
        return extract(clean_html(str(soup))), all_tabs

    def generate_documents(self) -> List[Dict]:

        print('HELLO', self.collection_path)
        #exit()
        
        document_batch = []
        #f = open('/mnt/data/obqa/corpus/crag/all_urls.txt', 'a')
        #all_urls = f.readlines()
        all_urls = []
        #print('Existing urls', len(all_urls))
        with jsonlines.open(self.collection_path) as file:
            data = list(file.iter())

        for d_id, d in enumerate(data):
            for doc_id, sr in enumerate(d['search_results']):
                if sr['page_url'] in all_urls:
                    print('skip duplicate...')
                    continue
                if sr['page_url'] in SKIP_URLS: 
                    print('skip url:', title)
                    continue
                print(sr['page_url'])
                title = sr['page_name']
                print(title)
                text, tables = self.get_page_text(sr['page_result'])
                if not text:
                    print('skip empty doc...')
                    continue
                parsed_document = {
                    "id": 'CRAG_' + str(d_id) + '_' + str(doc_id),
                    "url": sr['page_url'],
                    "title": title,
                    "contents": text
                }

                document_batch.append(parsed_document)
                all_urls.append(sr['page_url'])
                #f.write(sr['page_url'])

                if len(document_batch) == self.batch_size:
                    yield document_batch
                    document_batch = []
        
        if len(document_batch) > 0:
            yield document_batch


# cp /mnt/data/algomo/TIRESRAG-R1/evaluation/FlashRAG/scripts/CRAG_generator.py /mnt/data/obqa/corpus/trec-cast-tools/corpus_processing/generators; cp /mnt/data/algomo/TIRESRAG-R1/evaluation/FlashRAG/scripts/main_trec.py /mnt/data/obqa/corpus/trec-cast-tools/corpus_processing

# vi /mnt/data/obqa/corpus/trec-cast-tools/corpus_processing/generators/__init__.py

# cd /mnt/data/obqa/corpus/trec-cast-tools/corpus_processing
# python3 main_trec.py --process_crag --output_type jsonlines --output_dir /mnt/data/obqa/corpus/crag --duplicates_file /mnt/data/obqa/corpus/crag/all_duplicates.txt --crag_collection /mnt/data/obqa/corpus/crag/crag_task_3_dev_v4/crag_task_3_dev_v4_0.jsonl