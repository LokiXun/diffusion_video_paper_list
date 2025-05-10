# encoding: utf-8
"""
Function: Create paper note in markdown template. And fill in the given paper info.
@author: LokiXun
@contact: 2682414501@qq.com
"""
import logging
import time
import datetime
import re
from pathlib import Path
from enum import Enum
from typing import Tuple

import arxiv

#from scholarly import scholarly

logging.basicConfig(level=logging.INFO)
download_base_path = Path(r"C:\Users\Loki\workspace").resolve()
assert download_base_path.exists(), f"download_base_path={download_base_path} not exists!"

# search url in paper summary
CodeUrlPattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@&+s]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
code_url_pattern = re.compile(CodeUrlPattern)

# get paper save_name
time_format = "%Y_%m"
pdf_appendix = ".pdf"
note_appendix = "_Note.md"
MonthAbbreviation = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')

ADDITIONAL_NO_SAVE_CHARACTER_SET = {"\"", "|", "\\", "/", ">", "<"}

NO_SAVE_CHARACTER_SET = {':', "!", "?", ".", ",", "~",
                         "：", "！", "？", "。", "，"
                         }
CHAR_TO_REPLACE = "-"

def translate_paper_title_str(title_str: str) -> str:
    """
    process the paper title to save as file
    1. replace space with `-`
    2. get rid of `:` or `/`

    :param title_str:
    :return: save-able str
    """
    # 0. strip space
    save_str: str = title_str.strip().replace(" ", CHAR_TO_REPLACE)

    # 1. remove special character
    for need_delete_char in NO_SAVE_CHARACTER_SET:
        save_str = save_str.replace(need_delete_char, CHAR_TO_REPLACE)

    return save_str

def get_markdown_template_str(title: str, conference_journal_name: str, published_datetime: datetime.datetime,
                              paper_arxiv_url: str, code_url: str, paper_pdf_save_name: str,
                              authors_str: str, note_save_path:str="" ) -> str:
    """Markdown note template"""
    note_content_template = f"## Key-point\n\n" \
                        f"- Task\n" \
                        f"- Problems\n" \
                        f"- :label: Label:\n\n" \
                        f"## Contributions\n\n" \
                        f"## Introduction\n\n" \
                        f"## methods\n\n" \
                        f"## setting\n\n" \
                        f"## Experiment\n\n" \
                        f"> ablation study 看那个模块有效，总结一下\n\n" \
                        f"## Limitations\n\n" \
                        f"## Summary :star2:\n\n" \
                        f"> learn what\n\n" \
                        f"### how to apply to our task\n\n"
    
    return f"# {title}\n\n" \
           f"> \"{title}\" {conference_journal_name}, {published_datetime.year} {MonthAbbreviation[published_datetime.month - 1]} {published_datetime.day}\n" \
           f"> [paper]({paper_arxiv_url}) [code]({code_url}) [pdf](./{paper_pdf_save_name}) [note](./{note_save_path.name})\n" \
           f"> Authors: {authors_str}\n\n" \
           f"{note_content_template}"


class PublicationName(str, Enum):
    arxiv: str = "Arxiv"
    # conference
    aaai: str = "AAAI"
    emnlp: str = "EMNLP"
    icml: str = "ICML"
    iclr: str = "ICLR"
    nips: str = "NIPS"
    cvpr: str = "CVPR"
    cvpr_oral: str = "CVPR-oral"
    cvpr_high: str = "CVPR-highlight"
    cvpr_workshop: str = "CVPR-workshop"
    cvpr_best: str = "CVPR-bestpaper"
    siggraph: str = "SIGGRAPH"
    siggraph_asia: str = "SIGGRAPH-ASIA"
    eccv: str = "ECCV"
    eccv_workshop = "ECCV-workshop"
    iccv: str = "ICCV"
    wacv: str = "WACV"
    wacv_oral: str = "WACV-oral"
    icra: str = "ICRA"
    
    # journal
    acm_kdd: str = "ACM-KDD"
    acm_mm: str = "ACM-MM"
    acm_mm_oral: str = "ACM-MM-oral"
    optical_express: str = "OpticsExpress"
    tpami: str = "TPAMI"
    tip: str = "TIP"
    
    ichpc: str = "ICHPC"  # The International Conference for High Performance Computing, Networking, Storage and Analysis
    usenix: str = "USENIX"
    Neurocomputing: str = "Neurocomputing"


class PaperNotePreparation:

    @staticmethod
    def get_code_url(result: arxiv.Result,) -> str:
        """currently extract code url from summary
        TODO: search code on Github, reference https://blog.csdn.net/Next_Second/article/details/78238328
        """
        code_summary_str = result.summary

        code_url_list = re.findall(code_url_pattern, code_summary_str)
        print(f"code_url_list={code_url_list}")

        code_url = code_url_list[0] if code_url_list else ""
        if not code_url_list:
            print(f"Not find code url in summary! summary={code_summary_str}")
            code_url = ""

        return code_url

    @staticmethod
    def search_paper_publication_with_googleScholar(paper_name: str) -> Tuple[bool, str]:
        try:
            print(f">>>> start search_paper_publication_with_googleScholar:")
            search_results = scholarly.search_pubs(paper_name)
            one_result = next(search_results)
            find_pub = one_result['bib'].get('venue', None)
            
            return True, find_pub
        except Exception as e:
            print(e)
            return False, ""
    
    @staticmethod
    def get_where_to_publish(result: arxiv.Result) -> str:
        """TODO:find which conference or journal the paper published, default Arxiv"""
        conference_journal_name = result.journal_ref  # "Arxiv"
        conference_journal_name = conference_journal_name if conference_journal_name else "Arxiv"

        return conference_journal_name
    
    @staticmethod
    def create_markdown_note(title: str, conference_journal_name: str, published_datetime: datetime.datetime,
                            paper_arxiv_url: str, code_url: str, paper_pdf_save_name: str,
                            authors_str: str, note_save_path: Path):
        """
        create
        :return:
        """
        note_template_str = get_markdown_template_str(
            title=title, conference_journal_name=conference_journal_name, published_datetime=published_datetime,
            paper_arxiv_url=paper_arxiv_url, code_url=code_url, paper_pdf_save_name=paper_pdf_save_name,
            authors_str=authors_str, note_save_path=note_save_path)
        if note_save_path.exists():
            raise Exception(f"note already exists! path={note_save_path.as_posix()}")

        with open(note_save_path, 'w+',encoding='u8') as fp:
            fp.write(note_template_str)
        print(f"save markdown success! path={note_save_path}")

    @staticmethod
    def download_paper_pdf_arxiv(result: arxiv.Result, paper_pdf_save_name: str):
        written_path = None
        start_time: datetime.datetime = datetime.datetime.now()
        print(f"start download papepr pdf! start={start_time.strftime('%Y-%m-%d %X')}")
        try:
            written_path = result.download_pdf(dirpath=download_base_path.as_posix(), filename=paper_pdf_save_name)
        except Exception as e:
            logging.exception(f"download paper pdf failed! path="
                            f"{download_base_path.joinpath(paper_pdf_save_name).as_posix()}, written_path={written_path}")
            # delete tmp file
            pass
            print(f"no delete pdf tmp file yet!")
        print(f"download success! costs={(datetime.datetime.now() - start_time).seconds / 60}min, "
            f"path={download_base_path.joinpath(paper_pdf_save_name).as_posix()}, written_path={written_path}")


    # __call__
    def rephrase_given_paper_info_save_note(self, result: arxiv.Result, create_note_flag: bool = True, download_pdf_flag: bool = True, publication_name: str = ""):
        """
        Given paper info, rephrase paper info and create paper note in mardown template. Finally save locally.
        :param result:
        :return:
        """
        # 0. rephrase paper info
        reparsed_title = translate_paper_title_str(result.title)
        # additional not support symbol
        for need_delete_char in ADDITIONAL_NO_SAVE_CHARACTER_SET:
            reparsed_title = reparsed_title.replace(need_delete_char, CHAR_TO_REPLACE)

        published_datetime: datetime = result.published
        published_time = published_datetime.strftime(time_format)
        authors_str = ", ".join([a.name for a in result.authors])
        # find where to publish
        conference_journal_name = self.get_where_to_publish(result=result)
        # get paper url
        paper_arxiv_url = result.entry_id
        # paper_pdf_url = result.pdf_url
        code_url = self.get_code_url(result=result)

        # save name
        paper_save_name = f"{published_time}_{conference_journal_name}_{reparsed_title}"
        paper_pdf_save_name = f"{paper_save_name}{pdf_appendix}"
        note_save_name = f"{paper_save_name}{note_appendix}"
        note_save_path = download_base_path.joinpath(note_save_name)
        print(f"paper_save_name={paper_save_name} \n authors={authors_str}")

        # 1. create markdown
        print(result.title)
        if create_note_flag:
            self.create_markdown_note(title=result.title, conference_journal_name=conference_journal_name,
                                published_datetime=published_datetime,
                                paper_arxiv_url=paper_arxiv_url, code_url=code_url, authors_str=authors_str,
                                paper_pdf_save_name=paper_pdf_save_name, note_save_path=note_save_path)

        # 2. download paper pdf
        if download_pdf_flag:
            self.download_paper_pdf_arxiv(result, paper_pdf_save_name=paper_pdf_save_name)

        return True, download_base_path.joinpath(paper_pdf_save_name).as_posix(), note_save_path

    def prepare_paper_pdf_note_with_arxivIDorNAME(self, query_paper_name: str, arxiv_id_list: list = [], max_retry: int = 10,
                            create_note_flag=True, download_pdf_flag=True, strict=False, publication_name=None):
        """
        given arxiv paper url -> prepare a local mardown for paper reading
        """
        if not arxiv_id_list:
            assert query_paper_name, f"paper_name={paper_name} invalid!"

        # 0. search paper info in arixv website
        arxiv_search = arxiv.Search(
            query=query_paper_name,
            id_list=arxiv_id_list,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        big_slow_client = arxiv.Client(
            page_size=1000,
            delay_seconds=10,
            num_retries=5
        )
        # repeat to get paper info:
        result = None
        for i in range(max_retry):
            result_generator = big_slow_client.results(search=arxiv_search)
            try:
                print(f"Try {i} times: query={arxiv_search.query}")
                found_flag = False
                result = None
                while True:
                    result = next(arxiv_search.results())
                    print(f"    found {result.title}")
                    search_name = arxiv_search.query.replace("-", " ").replace(":", " ").lower()
                    found_name = str(result.title).replace("-", " ").replace(":", " ").lower()
                    if strict and search_name == found_name:
                        found_flag = True
                        break

                    if not strict and search_name in found_name:
                        found_flag = True
                        break
                if found_flag:
                    break
            except StopIteration as e:
                print(f"Try {i} times failed:{len(list(result_generator))}")
                continue
            time.sleep(1)

        if not result:
            raise Exception(f"search.max_results={arxiv_search.max_results}: Paper Not exists on arxiv, name={paper_name}")
        print(f"query={query_paper_name}, search.max_results={arxiv_search.max_results}")
        print(repr(result))
        # update publication_name if having one
        result.journal_ref = publication_name if publication_name else result.journal_ref
        
        
        # 1. fill the searched info into paper template and Save Note File
        self.rephrase_given_paper_info_save_note(result=result, create_note_flag=create_note_flag, download_pdf_flag=download_pdf_flag)


class TestPaperDownload(object):

    @staticmethod
    def test_create_markdown():
        # test download
        result = arxiv.Result(
            entry_id='https://www.sciencedirect.com/science/article/pii/S0143816621001779',
            updated=datetime.datetime(2021, 11, 1,tzinfo=datetime.timezone.utc),
            published=datetime.datetime(2021, 11, 1, tzinfo=datetime.timezone.utc),
            title="""RestoreNet-Plus: Image restoration via deep learning in optical synthetic aperture imaging system""",
            authors=[arxiv.Result.Author(x) for x in "Ju Tang, Ji Wu, Kaiqiang Wang, Zhenbo Ren, Xiaoyan Wu, Liusen Hu, Jianglei Di, Guodong Liu, Jianlin Zhao".split(", ") if x], # [arxiv.Result.Author('Justin Johnson'), arxiv.Result.Author('Alexandre Alahi'), arxiv.Result.Author('Li Fei-Fei')],
            summary="The code is\navailable at ",
            comment="",
            journal_ref="OpticsandLasers", #PublicationName.wacv.value,  # PublicationName.tip.value  
            doi="",
            primary_category="cs",
            categories=['cs.CV'],
            # links=[
            #     arxiv.Result.Link('http://arxiv.org/abs/2308.15070v1', title=None, rel='alternate', content_type=None),
            #     arxiv.Result.Link('http://arxiv.org/pdf/2308.15070v1', title='pdf', rel='related', content_type=None)],
            _raw=None,
        )

        _, pdf_path, note_path = PaperNotePreparation().rephrase_given_paper_info_save_note(result=result, create_note_flag=True, download_pdf_flag=False)


if __name__ == "__main__":
    paper_name = " "
    _arxiv_id_list = ["2501.00103v1"]
    search_strict_flag = False if _arxiv_id_list else False

    # manually type in publiction conference/journal's name
    _publication_name = PublicationName.arxiv.value
    # _publication_name = "Neurocomputing"
    # {'CVPR-highlight', 'AAAI', 'NIPS', 'ICLR', 'Arxiv', 'CVPR-workshop', 'ACM-KDD', 'OpticsExpress', 'CVPR', 'CVPR-bestpaper', 'ICML', 'EMNLP'}
    # supported_publications = set([x.value for x in PublicationName])
    # assert _publication_name in supported_publications
    _publication_name = _publication_name.strip(" ")


    PaperNotePreparation().prepare_paper_pdf_note_with_arxivIDorNAME(
        query_paper_name=paper_name, arxiv_id_list=_arxiv_id_list,
        strict=search_strict_flag, publication_name=_publication_name,
        # save choice
        create_note_flag=True,
        download_pdf_flag=False,
        )

    # # custom paper note, write info into Test
    # TestPaperDownload.test_create_markdown()
