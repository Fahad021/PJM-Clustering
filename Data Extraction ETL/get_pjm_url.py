import requests

from get_subscription_headers import get_subsription_headers

# create header with subscription key
headers = get_subsription_headers()


def get_url(name, links_json):
    return next(
        (
            link["links"][0]["href"]
            for link in links_json["items"]
            if name in link["name"]
        ),
        None,
    )


def get_pjm_links():
    url = "https://api.pjm.com/api/v1/"
    # fetch data at URL
    response = requests.get(url, headers=headers)

    return response.json()


def get_pjm_list():
    return get_pjm_links()["items"]


def get_pjm_url(name):
    links = get_pjm_links()

    return get_url(name, links)
