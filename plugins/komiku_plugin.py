from bs4 import BeautifulSoup
import re

class KomikuCoId:
    site = "https://komiku.co.id/"
    download_format = "pdf"

    def search(self, query: str, page: int) -> list:
        items = []
        with self.session.get(f"https://komiku.co.id/page/{page}/?post_type=manga&s={query}") as res:
            soup = BeautifulSoup(res.text, "html.parser")
            for d in [bge.find("a").attrs for bge in soup.findAll("div", class_="bge")]:
                if (url := d.get("href")):
                    items.append((self.manga(url).replace("-", " "), url))
            if not soup.find(class_="next"):
                return False, items
        return True, items

    def is_valid_url(self, url: str):
        return re.search(r"^https://komiku.co.id/manga/[^/]+", url)

    def chapters(self, url: str):
        chapters = []
        with self.session.get(url) as res:
            soup = BeautifulSoup(res.text, "html.parser")
            for link in soup.findAll("td", class_="judulseries")[::-1]:
                if (link := link.find("a").get("href")):
                    chapters.append(link)
        return chapters

    def images(self, url: str):
        with self.session.get(url) as res:
            soup = BeautifulSoup(res.text, "html.parser")
            for link in soup.findAll("img"):
                if (imgurl := link.get("src", "")).startswith("https://i0.wp.com/cdn.komiku.co.id"):
                     yield imgurl

    def manga(self, url: str) -> dict:
        if (re_ := re.search(r"^https://komiku.co.id/manga/([^/]+)", url)):
            return re_.group(1)

    def chapter(self, url: str):
        re_ = re.search(r"https://komiku.co.id/.*?chapter-([^/]+)", url)
        return re_.group(1).strip("-")
