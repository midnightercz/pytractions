import datetime
from typing import Optional, Type
import json
from jinja2 import Environment, FileSystemLoader

from ..models.gcrp_models import PR
from ..resources.gcrp_clients import GitHubClient, GitLabClient
from ..resources.json_store import JSONStore

from pytraction.base import Base, Traction, TList, In, Out, Res, Arg


class PRsModel(Base):
    authored_prs: TList[PR]
    for_review_prs: TList[PR]
    last_updated: str


class FetchPRs(Traction):
    r_ghclient: Res[GitHubClient]
    r_glclient: Res[GitLabClient]
    i_since: In[Optional[str]]
    o_authored_prs: Out[TList[Out[PR]]]
    o_for_review_prs: Out[TList[Out[PR]]]
    o_last_updated: Out[str]

    d_: str = """Fetch active pull requests from GitHub and GitLab.
    Gitlab and github are queried for PRs where user is author or requested for review.
    If i_since input is provided, only PRs which where updtaed after i_since datetime will be returned.
    Traction store results into o_authored_prs and o_for_review_prs outputs which
    combines both GitHub and GitLab PRs.
    """

    d_i_since: str = "Time used as last_updated parameter for PRs fetch. Expected format: YYYY-MM-DDTHH:MM:SS."
    d_o_authored_prs: str = "List of authored PRs."
    d_o_for_review_prs: str = "List of PRs where user is requested for review."
    d_o_last_updated: str = "Timestamp of last PR fetch. Basicaly returns current time in ISO 8601 format."

    def _run(self, on_update):
        gh_prs_authored = self.r_ghclient.r.authored_active_prs(since=self.i_since.data)
        gl_prs_authored = self.r_glclient.r.authored_active_prs(since=self.i_since.data)

        gh_prs_for_review = self.r_ghclient.r.review_requested_active_prs(since=self.i_since.data)
        gl_prs_for_review = self.r_glclient.r.review_requested_active_prs(since=self.i_since.data)

        self.o_authored_prs.data = TList[Out[PR]]([
                                    Out[PR](data = x) for x in gh_prs_authored + gl_prs_authored])
        self.o_for_review_prs.data = TList[Out[PR]]([
                                    Out[PR](data = x) for x in gh_prs_for_review + gl_prs_for_review])

        self.o_last_updated.data = datetime.datetime.now().isoformat()


class MakePRsModel(Traction):
    i_last_update: In[Optional[str]]
    i_authored_prs: In[TList[In[PR]]]
    i_for_review_prs: In[TList[In[PR]]]
    o_prs_model: Out[PRsModel] = Out[PRsModel](data=PRsModel(authored_prs=TList[PR](), for_review_prs=TList[PR](), last_updated=""))

    d_: str = """Combine provided inputs to one model."""
    d_i_last_update: str = "Last updated since to fetch PRs for. Expected format: YYYY-MM-DDTHH:MM:SS."
    d_i_authored_prs: str = "List of authored PRs."
    d_i_for_review_prs: str = "List of PRs where user is requested for review."
    d_o_prs_model: str = "Output model of PRs."



    def _run(self, on_update):
        model = PRsModel(
            authored_prs=TList[PR]([x.data for x in self.i_authored_prs.data]),
            for_review_prs=TList[PR]([x.data for x in self.i_for_review_prs.data]),
            last_updated=datetime.datetime.now().isoformat(),
        )
        self.o_prs_model.data = model


class StorePRs(Traction):
    i_prs_model: In[PRsModel]
    r_json_store: Res[JSONStore[PRsModel]]

    d_i_prs_model: str = "Input model of PRs to store"

    def _run(self, on_update):
        self.r_json_store.r.store(self.i_prs_model.data)


class RenderReport(Traction):
    i_prs: In[PRsModel]
    a_output_file: Arg[str]

    d_: str = """Render provided PRs model into HTML report.
    Structure of the report consists of two tree like structures, one for list of authored PRs
    and second one for PRs where user is requested for a review.
    Report is stored into a_output_file argument.
    """
    d_i_prs: str = "Input model of PRs use to render the report."
    d_a_output_file: str = "Output file path to store the report."


    def _render_tree_to_html(self, node, template, level=0):
        if isinstance(node, dict):
            template.append("<ul class='tree'>")
            for key, value in node.items():
                template.append(f"<li>{key}")
                self._render_tree_to_html(value, template, level + 1)
                template.append("</li>")
            template.append("</ul>")

        elif isinstance(node, list):
            for item in node:
                template.append("<li>")
                self._render_tree_to_html(item, template, level + 1)
                template.append("</li>")
        else:
            template.append("" * level + f": {node}")

    def render_json_tree_to_html(self, data, output_path):
        # Create a generic template
        template = []
        template.append("""
            <!DOCTYPE html>
            <html>
              <style>

                .tree{
                  --spacing : 2.0rem;
                  --radius  : 10px;
                }

                .tree li{
                  display      : block;
                  position     : relative;
                  padding-left : calc(2 * var(--spacing) - var(--radius) - 2px);
                  padding-top: 10px;
                  padding-bottom: 10px;
                }


                .tree ul{
                  margin-left  : calc(var(--radius) - var(--spacing));
                  padding-left : 0;
                }

                .tree ul li{
                  border-left : 2px solid #ddd;
                }

                .tree ul li:last-child{
                  border-color : transparent;
                }

                .tree ul li::before{
                  content      : '';
                  display      : block;
                  position     : absolute;
                  top          : calc(var(--spacing) / -2);
                  left         : -2px;
                  width        : calc(var(--spacing) + 2px);
                  height       : calc(var(--spacing) + 1px);
                  border       : solid #ddd;
                  border-width : 0 0 2px 2px;
                }

                .tree li.parent::after{
                  content       : '+';
                  background    : #ddd;
                  border-radius : 50%;
                }
                
                .tree li::after{
                  content       : '';
                  text-align    : center;
                  display       : block;
                  position      : absolute;
                  top           : calc(var(--spacing) / 2 - var(--radius));
                  left          : calc(var(--spacing) - var(--radius) - 1px);
                  width         : calc(2 * var(--radius));
                  height        : calc(2 * var(--radius));
                  border-radius : 0%;
                  background    : #777;
                }
                

                .tree ul.collapsed {
                  display: none;
                }
                .tree li.collapsed li {
                  display: none;
                }

                .fileitem {
                  border: 1px solid black;
                  background    : #777;
                  padding: 1px;
                  margin: 5px;
                  position      : relative;
                  display       : inner;
                }

                .spantext {
                  margin: 5px;
                }

              </style>
            <head>
                <title>PRs Report</title>
            </head>
            <body>
                <ul>
        """)

        # Render tree-like JSON to HTML
        self._render_tree_to_html(data, template)

        template.append("""
                </ul>
            </body>
            </html>
        """)

        # Save rendered HTML to file
        with open(self.a_output_file.a, 'w') as output_file:
            output_file.write("\n".join(template))

    def _run(self, on_update):
        self.render_json_tree_to_html(self.i_prs.data.content_to_json(), self.a_output_file.a)
