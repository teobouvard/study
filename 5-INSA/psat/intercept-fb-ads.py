"""Send a reply from the proxy without sending any data to the remote server."""
import json

from bs4 import BeautifulSoup
from mitmproxy import http

OUTPUT_DIR = "posts"


def response(flow: http.HTTPFlow) -> None:
    if flow.request.pretty_url.startswith("https://m.facebook.com/stories.php"):
        decoded_content = flow.response.content.decode("utf-8")
        trimmed_response = decoded_content.replace("for (;;);", "")
        json_response = json.loads(trimmed_response)
        html_part = json_response["payload"]["actions"][0]["html"]

        sponsored_posts = []
        for article in BeautifulSoup(html_part, features="lxml").find_all("article"):
            tokens = article.findAll(text=True)
            if "Sponsored" in tokens:
                sponsored_posts.append({"id": article["id"], "tokens": tokens})

        if sponsored_posts:
            filename = f"{OUTPUT_DIR}/{flow.request.timestamp_end}.json"
            with open(filename, "w") as f:
                json.dump(sponsored_posts, f, ensure_ascii=False)

