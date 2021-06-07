import html2text
import os
import re
import requests
import lib.config as config
from clean_plain_text import postprocess_markdown
from bs4 import BeautifulSoup
from lib.utils.csv_utils import read_csv_to_dict, save_dict_to_csv
from lib.utils.google_utils import GoogleService


def file_exist(folder, file_name):
    return os.path.exists("{}/{}".format(folder, file_name))


class DownloadBenchmark():
    def __init__(self):
        self.benchmark_folder = config.BENCHMARK_INFO_FOLDER
        self.benchmark_link = config.BENCHMARK_SHEET
        self.gs = GoogleService()
        self.wks = self.gs.open_file(self.benchmark_link)

    def download(self, domain, verbose=True):
        if verbose:
            print('Downloading {}'.format(domain))
        curr_wks = self.wks.worksheet(domain)
        save_dict_to_csv("{}/{}.csv".format(
            self.benchmark_folder, domain), curr_wks.get_all_records())

    def download_all(self, verbose=True):
        for curr_wks in self.wks.worksheets():
            self.download(curr_wks.title, verbose=verbose)

    def get(self, domain, offline=True, refresh=False):
        path = "{}/{}.csv".format(self.benchmark_folder, domain)
        if offline:
            if os.path.exists(path) and not refresh:
                return read_csv_to_dict(path)
            else:
                self.download(domain)
                return read_csv_to_dict(path)
        else:
            curr_wks = self.wks.worksheet(domain)
            return curr_wks.get_all_records()


class ProcessBenchmark():
    def __init__(self):
        self.db = DownloadBenchmark()

        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        self.h2t.ignore_emphasis = True
        self.h2t.ignore_tables = True
        self.h2t.ignore_images = True
        self.h2t.body_width = 512

    def replace_year_acro(self, text):
        text = re.sub(r"[']([0-2][0-9])", r" 20\1", text)
        text = re.sub(r"[']([3-9][0-9])", r" 19\1", text)

        text = re.sub(r"(^|[^\w\d])(Su)([0-2][0-9])", r"\1Summer 20\3", text)
        text = re.sub(r"(^|[^\w\d])(S|Sp)([0-2][0-9])", r"\1Spring 20\3", text)
        text = re.sub(r"(^|[^\w\d])(W|Wi)([0-2][0-9])", r"\1Winter 20\3", text)
        text = re.sub(r"(^|[^\w\d])(F|Fa)([0-2][0-9])", r"\1Fall 20\3", text)
        text = re.sub(r"(^|[^\w\d])(Au)([0-2][0-9])", r"\1Autumn 20\3", text)

        text = re.sub(r"(^|[^\w\d])(Su)([3-9][0-9])", r"\1Summer 19\3", text)
        text = re.sub(r"(^|[^\w\d])(S|Sp)([3-9][0-9])", r"\1Spring 19\3", text)
        text = re.sub(r"(^|[^\w\d])(W|Wi)([3-9][0-9])", r"\1Winter 19\3", text)
        text = re.sub(r"(^|[^\w\d])(F|Fa)([3-9][0-9])", r"\1Fall 19\3", text)
        text = re.sub(r"(^|[^\w\d])(Au)([3-9][0-9])", r"\1Autumn 10\3", text)

        return text

    def replace(self, text):
        replace_candidates_str = [
            ("Ph.D", "PhD"), ("Ph.D.", "PhD"), ("PhD.", "PhD"),
            ("Ph. D.", "PhD"), ("spring", "Spring"),
            ("fall", "Fall"), ("winter", "Winter"), ("summer", "Summer")
        ]

        for (orig, rep) in replace_candidates_str:
            text = text.replace(orig, rep)

        text = self.replace_year_acro(text)

        return text

    def request_url(self, bname, url):

        benchmark_utf16 = ["class_8", "class_9"]
        x = requests.get(url)

        if bname in benchmark_utf16:
            x.encoding = "UTF-16"

        return x.text

    def html2string_dumb(self, html_text):
        soup = BeautifulSoup(html_text, parser='lxml')
        for script in soup(["script", "style"]):
            script.extract()    # rip it out
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip()
                  for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text

    def html2string(self, html_text):
        md_text = self.h2t.handle(html_text)
        soup = BeautifulSoup(md_text, parser='lxml')
        text = "".join(soup.findAll(text=True))
        return text

    def save_file(self, folder, file_name, content):
        with open("{}/{}".format(folder, file_name), "w") as output_file:
            output_file.write(content)

    def read_file(self, folder, file_name):
        with open("{}/{}".format(folder, file_name), "r") as input_file:
            content = input_file.read()
            return content

    def read_benchmarks(self, domain, offline=True):
        return self.db.get(domain, offline)

    def merge_html(self, htmls):
        main_html = htmls[0]

        for sub_html in htmls[1:]:
            new_tag = sub_html.body
            new_tag.name = "div"

            filtered_tags = ["script", "nav"]
            for tag in filtered_tags:
                candidates = new_tag.findAll(tag)
                for c in candidates:
                    c.decompose()
            main_html.body.append(new_tag)

        return main_html.__repr__()

    def save_all_webpages(self, domain, offline=True, refresh=True):
        records = self.db.get(domain, offline, refresh)
        relevant_web_keys = [key for key in records[0].keys() if "web" in key]

        save_folder = "raw_htmls/{}".format(domain)

        for rec in records:
            _id = rec['id']

            if not _id:
                continue
            print("Processing", _id)

            for web_key in relevant_web_keys:

                # print(rec['include_main'] == 'FALSE')

                if domain == 'clinic' and rec['include_main'] == 'FALSE' and "main" in \
                        web_key:
                    continue

                if domain == "conference" and "main" in web_key:
                    continue

                url = rec[web_key]

                if url == "":
                    continue

                topic = web_key.split("_")[1]
                save_id = "{}-{}".format(_id, topic)

                if "|" in url:
                    split_url = url.split("|")
                    for i in range(len(split_url)):
                        save_id_i = "{}-{}".format(save_id, str(i))
                        if not file_exist(save_folder, save_id_i + '.html'):
                            su = split_url[i]
                            html_text = self.request_url(_id, su)

                            self.save_file(save_folder, save_id_i + '.html',
                                           html_text)
                else:
                    if not file_exist(save_folder, save_id + '.html'):
                        html_text = self.request_url(_id, url)

                        self.save_file(save_folder, save_id + '.html',
                                       html_text)

    # handle fac_42, fac_20 header issue
    def instrutment_bs(self, content):
        dom = BeautifulSoup(content, parser="lxml")
        # find <h1><em>
        title_id = ["h1"]
        for tag in title_id:
            has_title = False
            title = None
            not_title_follows_title = []
            for ele in dom.find_all(tag):
                title_em = ele.find("em")
                if title_em is not None:
                    has_title = True
                    continue
                title_center = ele.find("center")
                if title_center is not None:
                    has_title = True
                    # NOTE: if i don't do this fac_20 cannot parse the header
                    title_center.unwrap()
                    continue
                if has_title:
                    not_title_follows_title.append(ele)

            if has_title:
                for tag in not_title_follows_title:
                    tag.name = "h2"
        return dom

    def save_benchmarks_realted_info(
            self, domain, offline=True, refresh=True, b=None, append=True, append_key=['sponsor']):

        read_folder = "raw_htmls/{}".format(domain)

        records = self.db.get(domain, offline, refresh)
        relevant_web_keys = [key for key in records[0].keys() if "web" in key]

        for rec in records:
            _id = rec['id']

            if not _id:
                continue

            if b is not None and not "{}_{}".format(domain, b) == _id:
                continue

            print("Processing", _id)

            htmls = []
            for web_key in relevant_web_keys:

                if domain == 'clinic' and rec['include_main'] == 'FALSE' and "main" in \
                        web_key:
                    continue

                if domain == "conference" and "main" in web_key:
                    continue

                url = rec[web_key]

                if url == "":
                    continue

                topic = web_key.split("_")[1]
                save_id = "{}-{}".format(_id, topic)

                if "|" in url:
                    split_url = url.split("|")
                    for i in range(len(split_url)):
                        save_id_i = "{}-{}".format(save_id, str(i))
                        htmls.append(self.instrutment_bs(
                            self.read_file(read_folder, save_id_i + '.html')))
                else:
                    htmls.append(self.instrutment_bs(
                        self.read_file(read_folder, save_id + '.html')))

            html_text = self.merge_html(htmls)

            html_text = self.replace(html_text)

            self.save_file("raw_htmls", _id + '.html', html_text)

            plain_text = postprocess_markdown(self.html2string(html_text))
            self.save_file("plain_texts", _id + '.txt', plain_text)


class ParseBenchmark:
    """
    Class to execute WebExtract_Parser on raw_htmls.
    """
    def __init__(self):
        pass

    def parse_all(self, benchmark=None):
        # run parser on raw_htmls
        benchmark_folder = os.path.join(
            os.getcwd(), config.RAW_BENCHMARK_FOLDER)
        benchmark_output = os.path.join(
            os.getcwd(), config.PARSED_BENCHMARK_FOLDER)

        command = \
            '{} --benchmark_folder {} --benchmark_output {} ' \
            .format(
                config.PARSER_EXECUTABLE,
                benchmark_folder,
                benchmark_output
            )
        if benchmark:
            command += '--benchmark {}'.format(benchmark)
        os.system(command)
        print(command)
