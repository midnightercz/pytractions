import json
from typing import Optional

from ..resources.gcrp_clients import GitHubClient, GitLabClient
from ..resources.json_store import JSONStore
from ..tractions.gcrp_tractions import FetchPRs, StorePRs, PRsModel, MakePRsModel, RenderReport

from pytraction.tractor import Tractor
from pytraction.base import Base, TList, In, Res, Arg


class GCRP_PR_Tractor(Tractor):
    i_since: In[Optional[str]] = In[Optional[str]](data=None)
    r_ghclient: Res[GitHubClient] = Res[GitHubClient](r=GitHubClient(user="fake", token="fake"))
    r_glclient: Res[GitLabClient] = Res[GitLabClient](r=GitLabClient(user="fake", token="fake", host="example.com"))
    r_json_store: Res[JSONStore] = Res[JSONStore[PRsModel]](r=JSONStore(filename="fake.json"))
    a_report_file: Arg[str] = Arg[str](a="report.html")

    t_fetch_prs: FetchPRs = FetchPRs(uid='fetch-1', 
                                     i_since=i_since,
                                     r_ghclient=r_ghclient, r_glclient=r_glclient)
    t_make_prs_model: MakePRsModel = MakePRsModel(uid='make-1',
                                                  i_last_update=t_fetch_prs.o_last_updated,
                                                  i_authored_prs=t_fetch_prs.o_authored_prs,
                                                  i_for_review_prs=t_fetch_prs.o_for_review_prs)
    t_store_prs: StorePRs = StorePRs(uid='store-1',
                                     i_prs_model=t_make_prs_model.o_prs_model,
                                     r_json_store=r_json_store)

    t_render_report: RenderReport = RenderReport(uid='render-1', i_prs=t_make_prs_model.o_prs_model,
                                                 a_output_file=a_report_file)


def main():
    t = GCRP_PR_Tractor(
        uid='gcrp-pr-1',
        r_ghclient=Res[GitHubClient](r=GitHubClient(user="midnightercz", token="")),
        r_glclient=Res[GitLabClient](r=GitLabClient(user="4158", token="", host="gitlab.cee.redhat.com")),
        r_json_store=Res[JSONStore](r=JSONStore(filename="gh_pr.json")),
        a_report_file=Arg[str](a="report.html")
    )
    t.run()
    print(json.dumps(t.to_json()))


def to_json():
    return GCRP_PR_Tractor.type_to_json()
