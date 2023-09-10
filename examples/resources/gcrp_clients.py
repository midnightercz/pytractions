import requests

from typing import List, Dict

from ..models.gcrp_models import PR
from pytraction.base import Base, TDict

class GitHubClient(Base):
    user: str
    token: str

    def user_orgs(self):
        headers = {
            'Authorization': f'Bearer {self.token}'
        }

        url = f'https://api.github.com/users/{self.user}/orgs'
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        orgs = response.json()
        return orgs

    def org_repositories(self, org_name):
        headers = {
            'Authorization': f'Bearer {self.token}'
        }
        url = f'https://api.github.com/orgs/{org_name}/repos'
        repositories = []

        while url:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            repos = response.json()
            repositories.extend(repos)

            if 'Link' in response.headers and 'rel="next"' in response.headers['Link']:
                url = response.headers['Link'].split(';')[0][1:-1]
            else:
                url = None

        return repositories

    def user_repositories(self):
        headers = {
            'Authorization': f'Bearer {self.token}'
        }
        url = f'https://api.github.com/users/{self.user}/repos'
        repositories = []

        while url:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            repos = response.json()
            repositories.extend(repos)

            if 'Link' in response.headers and 'rel="next"' in response.headers['Link']:
                url = response.headers['Link'].split(';')[0][1:-1]
            else:
                url = None

        return repositories

    def authored_active_prs(self, since=None) -> List[PR]:
        headers = {
            'Authorization': f'Bearer {self.token}'
        }
        url = f'https://api.github.com/search/issues?q=is:pr+author:{self.user}+is:open'
        if since:
            url += f'+since:{since}'

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        prs = response.json()['items']
        pr_list = []
        for pr in prs:
            pr_list.append(PR(
                link=pr['html_url'],
                title=pr['title'],
                author=pr['user']['login'],
                updated_at=pr['updated_at'],
                created_at=pr['created_at'],
                body=pr['body'] or ""
            ))
        return pr_list

    def review_requested_active_prs(self, since=None) -> List[PR]:
        headers = {
            'Authorization': f'Bearer {self.token}'
        }
        url = f'https://api.github.com/search/issues?q=is:pr+review-requested:{self.user}+is:open'
        if since:
            url += f'+since:{since}'

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        prs = response.json()['items']
        pr_list = []
        for pr in prs:
            pr_list.append(PR(
                link=pr['html_url'],
                title=pr['title'],
                author=pr['user']['login'],
                updated_at=pr['updated_at'],
                created_at=pr['created_at'],
                body=pr['body'] or ""
            ))
        return pr_list


class GitLabClient(Base):
    token: str
    user: str
    host: str

    @property
    def project_url(self):
        return f"https://{self.host}/api/v4/projects"

    @property
    def mrs_url(self):
        return f"https://{self.host}/api/v4/merge_requests"

    @property
    def todos_url(self):
        return f'https://{self.host}/api/v4/todos'

    def get_user_projects(self):
        self.headers = {
            'PRIVATE-TOKEN': self.token
        }
        params = {
            'membership': 'true',
            'archived': 'false'
        }
        response = requests.get(self.projects_url, headers=self.headers, params=params)

        if response.status_code == 200:
            projects = response.json()
            return projects
        else:
            print(f"Error getting projects: {response.status_code} - {response.text}")
            return []

    def authored_active_prs(self, since=None):
        headers = {
            'PRIVATE-TOKEN': self.token
        }
        page = 1
        while True:
            pr_list = []
            mr_url = self.mrs_url
            params = {
                'author_id': self.user,
                'state': 'opened',
                'page': page,
            }
            if since:
                params['update_after'] = since
            response = requests.get(mr_url, headers=headers, params=params)
            response.raise_for_status()
            #print('--')
            if not response.json():
                break
            for pr in response.json():
                pr_list.append(PR(
                    link=pr['web_url'],
                    title=pr['title'],
                    author=pr['author']['username'],
                    updated_at=pr['updated_at'],
                    created_at=pr['created_at'],
                    body=pr['description'] or ""
                ))
            page += 1
        return pr_list

    def review_requested_active_prs(self, since=None):
        page = 1
        per_page = 20  # Adjust this based on your preference and GitLab's rate limits
        params = {
            'action_id': '2',  # 2 corresponds to "created" action (i.e., assigned to you)
            'state': 'pending',
            'page': page,
            'target_type': 'MergeRequest',
            'per_page': per_page,
        }
        if since:
            params['update_after'] = since
        headers = {
            'PRIVATE-TOKEN': self.token
        }

        pr_list = []

        while True:
            response = requests.get(self.todos_url, headers=headers, params=params)
            response.raise_for_status()
            todos = response.json()
            if not todos:
                break  # No more todos to fetch
            for todo in todos:
                pr_list.append(PR(
                    link=todo['target']['web_url'],
                    title=todo['target']['title'],
                    author=todo['author']['username'],
                    updated_at=todo['target']['updated_at'],
                    created_at=todo['target']['created_at'],
                    body=todo['target']['description']
                ))
            page += 1
            params['page'] = page
        return pr_list
