from typing import List, Union, Optional, Any
from queue import Queue
from PIL import Image
from PyPDF2 import PdfFileReader, PdfFileWriter

import shutil
import curses
import curses.ascii
import cloudscraper
import _queue
import threading
import os
import itertools
import re
import time
import textwrap
import mimetypes
import os
import inspect
import importlib
import textwrap


# follow me: https://fb.com/zvtyrdt.id


banner = """
██╗  ██╗ ██████╗ █████╗ ███████╗████████╗██████╗ ██╗
██║ ██╔╝██╔════╝██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██║
█████╔╝ ██║     ███████║███████╗   ██║   ██║  ██║██║
██╔═██╗ ██║     ██╔══██║╚════██║   ██║   ██║  ██║██║
██║  ██╗╚██████╗██║  ██║███████║   ██║   ██████╔╝███████╗
╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═════╝ ╚══════╝ Tui 0.3
"""


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def progressBar(current: int, total: Optional[int] = None, *, decimals: int = 1,
                lenght: int = 20, progressLenght: int = 20, fill: str = "=") -> str:
    if not total:
        total = current
        if total == 0:
            total = 1
    elif not isinstance(total, int):
        total = int(total)
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (current / float(total)))
    filledLenght = int(lenght * current // total)
    bar = fill * filledLenght + '.' * (lenght - filledLenght - 1)
    current = sizeof_fmt(current)
    total = sizeof_fmt(total)
    progress = f"{{0:<{progressLenght}}}".format(f"{current}/{total}")
    return f"|{bar}| {percent}% | {progress}"


def load_plugins():
    klasses = {}
    required_funcs = ["search", "chapters", "manga", "chapter"]
    positional_funcs = ["pdfs", "images"]
    for pluginfile in os.listdir("plugins"):
        if re.search(r".+plugin\.py$", pluginfile):
            plugin_name = pluginfile[:-3]
            plugin = importlib.import_module(f"plugins.{plugin_name}")
            for name, klass in inspect.getmembers(plugin, inspect.isclass):
                if all(hasattr(klass, i) for i in required_funcs) and any(hasattr(klass, i) for i in positional_funcs):
                    klasses[name] = klass
    return klasses


class Wraptext:
    def __init__(self):
        self.text = ""
        self.width = 20
        self.maxline = 3
        self.index = 0
        self.maxindex = float("inf")
        self.prevdirect = None
        self._result = []

    def update(self, text: Union[str, list], width: int = None, maxline: int = None):
        if self.text != text:
            if isinstance(text, (list, tuple)):
                text = list(map(str, text))
            else:
                text = [str(text)]
            self.text = text
        if width and width != self.width:
            self.width = width
            self.maxindex = float("inf")
        if maxline and maxline != self.maxline:
            self.maxline = maxline
        self.next
        if self._result:
            self.back

    def _calc(self, back=False):
        if len(self.text) > 0:
            if back and self.index > 0:
                self.index -= 1
            if all(len(i) == 1 for i in self.text):
                wrapped = self.text
            else:
                wrapped = [tx for text in self.text for tx in textwrap.wrap(
                    text, self.width)]
            splited = [wrapped[i + self.index: i + self.index + self.maxline]
                       for i in range(0, len(wrapped), self.maxline)]
            result = splited[0]
            if len(result) < self.maxline:
                result = wrapped
            if result[-1] == wrapped[-1]:
                self.maxindex = self.index
            if not back and self.index < self.maxindex:
                self.index += 1
        else:
            result = []
        self._result = result
        return result

    def __iter__(self):
        return iter(self._result)

    def __repr__(self):
        return f"{self._result}"

    def __len__(self):
        return len(self._result)

    @property
    def next(self):
        if self.prevdirect == "back":
            self.index += 1
        self.prevdirect = "next"
        return self._calc()

    @property
    def back(self):
        if self.prevdirect == "next" and self.index != self.maxindex:
            self.index -= 1
        self.prevdirect = "back"
        return self._calc(back=True)


class Komik(object):
    def __init__(self, threadNum=5):
        self.plugins = load_plugins()
        self.threadNum = threadNum
        self.items = Queue()
        self.tempdata = {}
        self.images = {}

        self.session = cloudscraper.CloudScraper()
        self.block_size = 1024  # 1MB

    def startWrapper(self):
        curses.wrapper(self._wrapper)

    def download(self, index:  Any, url: str, output: str):
        r = self.session.head(url)
        file_size = int(r.headers.get('content-length', 0))
        content_type = r.headers.get('content-type', "image/jpeg")
        extension = mimetypes.guess_extension(content_type)
        output = f"{output}{extension}"

        resume_byte_pos = None
        downloaded = 0
        if os.path.isfile(output):
            local_size = os.stat(output).st_size
            if local_size != file_size:
                resume_byte_pos = local_size
                downloaded = local_size
        resume_header = ({'Range': f'bytes={resume_byte_pos}-'}
                         if resume_byte_pos else None)
        mode = "ab" if resume_byte_pos else "wb"
        with self.session.get(url, stream=True, headers=resume_header) as res:
            with open(output, mode) as out:
                for chunk in res.iter_content(self.block_size):
                    if chunk:
                        self.tempdata[index] = (downloaded, file_size, url)
                    downloaded += len(chunk)
                    out.write(chunk)
        return output

    def _worker(self, index: str) -> None:
        while 1:
            try:
                chapter, image_num, url = self.items.get(timeout=1)
            except _queue.Empty:
                self.tempdata[index] = None
                break

            dir = os.path.join("downloaded", self.source_manga,
                               self.manga_name, chapter)
            try:
                if not os.path.isdir(dir):
                    os.makedirs(dir)
            except Exception:
                pass
            output = os.path.join(dir, image_num)
            real_output = self.plugin.download(index, url, output)
            if not self.images.get(dir):
                self.images[dir] = []
            self.images[dir].append((real_output, url))
            self.items.task_done()

    def startWorker(self):
        for th in range(self.threadNum):
            newThread = threading.Thread(
                target=self._worker, args=(th,))
            newThread.setDaemon(True)
            self.tempdata[th] = ()
            newThread.start()

        # display
        index = self.printBanner()
        while any(item is not None for item in self.tempdata.values()):
            for i, item in self.tempdata.items():
                if item:
                    *pb, url = item
                    self.scr.addstr(i * 2 + index, 2, f"url: {url}")
                    self.scr.addstr(i * 2 + index + 1, 4,
                                    f"{progressBar(*pb)}")
            queue = self.items.qsize()
            self.scr.addstr(self.threadNum * 2 + 1 +
                            index, 2, f"queue: {queue}  ")
            self.scr.refresh()
        self.items.join()  # join all

        if self.is_download_image:
            # to pdf
            for out, files in self.images.items():
                xout = out + ".pdf"
                self.convertPdf(xout, *files)

    def input(self, ln: int, cl: int, msg: str, *, func=lambda x: x, finalFunc=lambda x: x, default=None) -> int:
        ch, res = 0, ''
        curses.curs_set(1)
        while True:
            if (dm := divmod(len(res) + len(msg), self.scr.getmaxyx()[1])[0]) > -1:
                x = dm or 1
            self.scr.addstr(ln, cl, f"{msg}: {res}")
            self.scr.refresh()
            ch = self.scr.getch()
            if ch in (curses.KEY_BACKSPACE, 127, ord("\b")):
                if res != "":
                    res = res[:-1]
                    self.scr.deleteln()
            elif ch == ord("\n"):
                if default and res == "":
                    res = default
                    break
                else:
                    if res != "" and finalFunc(res):
                        break
            elif curses.ascii.isascii(ch):
                if func(res + chr(ch)):
                    res += chr(ch)
        curses.curs_set(0)
        return ln + x, res

    def parseRange(self, raw: str, items: Optional[list] = [], max=float("inf"), inline_input: Optional[bool] = False):
        if (_re := re.match(r"^(\d+)?(\:?)(\d+)?$", raw)):
            start, dt, end = map(lambda x: int(x) if str(
                x).isdigit() else x, _re.groups())
            if len(items) == 0:
                if start == 0 or end == 0:
                    return False
                if start and end and not inline_input and end <= start:
                    return False
                if start and start > max:
                    return False
                if end and end > max:
                    return False
                return True
            else:
                if start:
                    start -= 1
                if not dt and end is None:
                    end = False
                if isinstance(start, int) and end:
                    return items[start:end]
                elif isinstance(start, int):
                    if end is None:
                        return items[start:]
                    else:
                        return [items[start]]
                elif end and start is None:
                    return items[:end]
                else:
                    return items[start:end]
        return False

    def printBanner(self):
        self.scr.clear()
        for index, line in enumerate(banner.splitlines(), start=2):
            self.scr.addstr(index, 2, line)
        if hasattr(self, "plugin"):
            self.scr.addstr(index + 1, 2, f"site: {self.plugin.site}")
            index += 1
        return index + 2

    def pdf_cat(self, input_files, output_stream):
        downdir, mangasite, manganame, chaptername = output_stream.split("/")
        index = self.printBanner()
        filename = "_".join([manganame, chaptername])
        fullpath = os.path.join(downdir, mangasite, manganame, filename)

        self.scr.addstr(index, 2, f"menyatukan file pdf -> {filename}")
        self.scr.refresh()

        input_streams = []
        try:
            # First open all the files, then produce the output file, and
            # finally close the input files. This is necessary because
            # the data isn't read from the input files until the write
            # operation. Thanks to
            # https://stackoverflow.com/questions/6773631/problem-with-closing-python-pypdf-writing-getting-a-valueerror-i-o-operation/6773733#6773733
            for input_file in input_files:
                input_streams.append(open(input_file, 'rb'))
                os.remove(input_file)
            writer = PdfFileWriter()
            for reader in map(PdfFileReader, input_streams):
                for n in range(reader.getNumPages()):
                    writer.addPage(reader.getPage(n))
            writer.write(open(fullpath, "wb"))
        finally:
            for f in input_streams:
                f.close()
            try:
                shutil.rmtree(output_stream[:-4])
            except Exception:
                pass
            self.scr.addstr(index + 1, 2, f"output: {fullpath}")
            self.scr.refresh()

    def convertPdf(self, output: str, *files: List[str]):
        import itertools
        index = self.printBanner()
        isdone = False

        def animate():
            n = itertools.cycle(r"\|/-")
            while not isdone:
                self.scr.addstr(
                    index, 2, f"converting {n.__next__()}")
                self.scr.refresh()
                time.sleep(0.3)

        th = threading.Thread(target=animate)
        th.setDaemon(True)
        th.start()

        # converting
        tmp_pdf = []
        files = sorted(files, key=lambda x: x[0])
        split_images = [files[i:i+5] for i in range(0, len(files), 5)]
        for num, files in enumerate(split_images):
            images = []
            output_tmp = output + f".tmp{num}"
            for file, url in files:
                try:
                    image = Image.open(file)
                except Exception as e:
                    self.scr.addstr(index + 1, 2, f"file korup ditemukan")
                    self.scr.refresh()
                    _, chapter, f = file.split("/")
                    self.items.put(
                        (chapter, f.split(".")[0], url)
                    )
                    continue
                image = image.convert("RGB")
                images.append(image)

            if self.items.qsize() > 0:
                isdone = True
                self.startWorker()
                continue

            self.scr.addstr(index + 1, 2, f"menyimpan file {output_tmp!r}    ")
            self.scr.refresh()
            if len(images) > 1:
                pdf, *images = images
                pdf.save(output_tmp, "PDF", save_all=True,
                         append_images=images)
            else:
                images[0].save(output_tmp, "PDF")
            tmp_pdf.append(output_tmp)
        isdone = True
        self.pdf_cat(tmp_pdf, output)

    def _get_direct_url(self, index: int, chapters: List[str], chapter_range: str):
        localQueue = Queue()

        def localWorker():
            while 1:
                try:
                    chapter_url = localQueue.get(timeout=1)
                except _queue.Empty:
                    break
                self.scr.addstr(
                    index + 2, 4, progressBar(totalChapter - localQueue.qsize(), totalChapter))
                self.scr.refresh()
                chapter = self.plugin.chapter(chapter_url)
                for num, imgurl in enumerate(self._method(chapter_url), start=1):
                    self.items.put((chapter, str(num), imgurl))
                localQueue.task_done()

        for chapter in chapters:
            localQueue.put(chapter)
        totalChapter = localQueue.qsize()
        if chapter_range.startswith(":"):
            chapter_range = "start" + chapter_range
        elif chapter_range.endswith(":"):
            chapter_range = chapter_range + "end"
        self.scr.addstr(
            index + 1, 2, f"mengunduh chapter: {chapter_range.replace(':', ' - ')}")

        for _ in range(2):
            th = threading.Thread(target=localWorker)
            th.setDaemon(True)
            th.start()
        localQueue.join()

    def _searchManga(self, query: str):
        all_manga = []
        selected = None

        def displayable(items):
            return [items[i:i + 10] for i in range(0, len(items), 10)]

        def localSearch(page):
            line = self.printBanner()
            self.scr.addstr(
                line, 2, f"mengunduh manga halaman {page}, kueri {query!r}")
            self.scr.refresh()

            maxpagereached, items = self.plugin.search(query, page)
            items = list(enumerate(items, start=len(all_manga) + 1))
            all_manga.extend(items)
            return not maxpagereached, items

        index = 0
        pagging = 0
        pagenum = 1

        maxPageReached, list_manga = localSearch(pagenum)
        items_to_display = displayable(all_manga)

        while not selected:
            line = self.printBanner()
            if not list_manga:
                self.scr.addstr(line, 2, f"manga tidak ditemukan!")
                self.scr.getch()
                return index, False

            width = self.scr.getmaxyx()[1] - 10
            pager = items_to_display[pagging]

            self.scr.addstr(
                line, 2, f"page {pagging + 1} dari {len(items_to_display)}, total manga {len(all_manga)}")
            for l, (num, (title, url)) in enumerate(pager, start=line + 2):
                self.scr.addstr(
                    l, 2, f"{num}. {textwrap.shorten(title, width)}")
            self.scr.refresh()

            start, end = pager[0][0], pager[-1][0]
            index, user = self.input(l + 2, 2, f"[k]embali,[l]anjut,[c]ari manga,[{start}-{end}]",
                                     func=lambda x: x in ["k", "l", "c"] or
                                     any(i.startswith(x)
                                         for i in map(str, range(start, end + 1))),
                                     finalFunc=lambda x: x in [
                                         "k", "l", "c", *map(str, range(start, end + 1))]
                                     )

            if user == "c":
                return index, False
            elif user == "k" and pagging > 0:
                pagging -= 1
            elif user == "l":
                if pagging < len(items_to_display) - 1:
                    pagging += 1
                elif not maxPageReached:
                    if pagging == len(items_to_display) - 1:
                        pagenum += 1
                        maxPageReached, list_manga = localSearch(pagenum)
            elif user.isdigit():
                rindex = [i[0] for i in pager].index(int(user))
                selected = pager[rindex][1][1]
        return -1, selected

    def select_manga_site(self, items: Union[list, dict]):
        if len(items) == 1:
            return items[0]
        selected = 0
        width = self.scr.getmaxyx()[1] - 5
        F = "{0:>%s}. {1:<%s}" % (len(str(len(items))), width)
        wraptext = Wraptext()
        wrap_items = [textwrap.shorten(
            F.format(i, f"{item}: {self.plugins[item].site}"), width)
            for i, item in enumerate(items, start=1)]
        wraptext.update(wrap_items, width=float("inf"), maxline=10)

        ln = self.printBanner()
        self.scr.addstr(
            ln, 2, f"pilih situs baca komik, total {len(wraptext)}")
        ln += 2

        while 1:
            for index, line in enumerate(wraptext, start=ln):
                if index == selected + ln:
                    self.scr.addstr(index, 4, line, curses.A_REVERSE)
                else:
                    self.scr.addstr(index, 4, line)
            self.scr.refresh()
            ch = self.scr.getch()

            if ch in (curses.KEY_DOWN, ord("s"), ord("S")):
                if selected < len(wraptext) - 1:
                    selected += 1
                else:
                    wraptext.next
            elif ch in (curses.KEY_UP, ord("w"), ord("W")):
                if selected > 0:
                    selected -= 1
                else:
                    wraptext.back
            elif ch == ord("\n"):
                return items[selected]

    def _loadPlugin(self, plugin: 'class'):
        self.plugin = plugin()
        self.plugin.session = self.session
        if not hasattr(self.plugin, "download"):
            self.plugin.download = self.download

    def _wrapper(self, scr) -> None:
        self.scr = scr
        curses.curs_set(0)
        curses.noecho()

        while 1:
            self.is_download_image = True
            self.source_manga = self.select_manga_site(
                list(self.plugins.keys()))
            self._loadPlugin(self.plugins[self.source_manga])

            url = False
            while 1:
                if not url:
                    index = self.printBanner()
                    index, url = self.input(index, 2, "Full Url/Search Query")
                elif not self.plugin.is_valid_url(url):
                    _, url = self._searchManga(url)
                else:
                    index = self.printBanner()
                    self.scr.addstr(index, 2, f"dipilih: {url}")
                    index += 1
                    break
            self.manga_name = self.plugin.manga(url).replace(" ", "-")
            self.scr.addstr(index, 2, "mengunduh semua chapter..")
            self.scr.refresh()
            chapters = self.plugin.chapters(url)
            self.scr.deleteln()

            lchaps = len(chapters)
            index, chapter_range = self.input(index, 2,
                                              "Chapter Range<int> (eg: 1,1:6,:8)"
                                              f" (default 1:) [1-{lchaps}]",
                                              func=lambda x: self.parseRange(
                                                  x, max=lchaps, inline_input=True),
                                              finalFunc=lambda x: self.parseRange(
                                                  x, max=lchaps),
                                              default="1:"
                                              )
            index, thread = self.input(index, 2, "Download Thread<int> [1-10] (default 5)",
                                       func=lambda x: True if x.isdigit() and int(x) > 0 and int(x) < 11 else False, default=5)
            methods = {}
            if hasattr(self.plugin, "pdfs"):
                methods["pdf"] = self.plugin.pdfs
            if hasattr(self.plugin, "images"):
                methods["image"] = self.plugin.images
            if len(methods) == 2:
                index, user = self.input(
                    index, 2, "Download Format [p]df atau [g]ambar", func=lambda x: x in ["p", "g"])
                selected_method = {"p": "pdf", "g": "image"}[user]
            else:
                selected_method = list(methods.keys())[0]
            if selected_method == "pdf":
                self.is_download_image = False
            self._method = methods[selected_method]

            chapters = self.parseRange(chapter_range, chapters)
            self._get_direct_url(index, chapters, chapter_range)

            # download threads
            if thread != "":
                self.threadNum = int(thread)
            if self.threadNum > (task := self.items.qsize()):
                self.threadNum = task
            self.startWorker()

            # wait
            _, user = self.input(index, 2, "selesai! tekan 'k' untuk keluar atau enter untuk cari manga baru",
                                 func=lambda x: x in ["k", "\n"],
                                 default="\n")
            if user == "k":
                break


komik = Komik()
komik.startWrapper()
