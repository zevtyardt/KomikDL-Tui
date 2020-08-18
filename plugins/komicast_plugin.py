from bs4 import BeautifulSoup
import re

class KomikCast:
    site = "https://komikcast.com"
    def search(self, query: str, page: int) -> list:
        items = []
        with self.session.get(f"https://komikcast.com/page/{page}/?s={query}") as res:
            soup = BeautifulSoup(res.text, "html.parser")
            for d in [bsx.find("a").attrs for bsx in soup.findAll("div", class_="bsx")]:
                if (title := d.get("title")) and (url := d.get("href")):
                    items.append((title, url))
            if not soup.find("a", class_="next page-numbers"):
                return False, items
        return True, items

    def is_valid_url(self, url: str):
        return re.search(r"^https://komikcast.com/komik/(.+?)", url)

    def chapters(self, url: str):
        chapters = []
        with self.session.get(url) as res:
            soup = BeautifulSoup(res.text, "html.parser")
            for link in soup.findAll("ul")[1].findAll("li")[::-1]:
                if (link := link.find("a").get("href")):
                    chapters.append(link)
        return chapters

    def images(self, url: str):
        with self.session.get(url) as res:
            soup = BeautifulSoup(res.text, "html.parser")
            for link in soup.findAll("img"):
                if (imgurl := link.get("src", "")).startswith("https://cdn.komikcast.com"):
                     yield imgurl

    def manga(self, url: str) -> dict:
        if (re_ := re.search(r"https://komikcast.com/komik/([^/]+)", url)):
            return re_.group(1)

    def chapter(self, url: str):
        re_ = re.search(r"https://komikcast.com/chapter/.*?chapter-([\d-]+)-?", url)
        return re_.group(1).strip("-")
