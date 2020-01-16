
from icrawler.builtin import GoogleImageCrawler
from datetime import date

# downloads all images from google search
def crawl(keyword: str, amount: int, folder: str=""):
    google_crawler = GoogleImageCrawler(
        parser_threads=2, 
        downloader_threads=4,
        storage={"root_dir": folder})

    google_crawler.crawl(
        keyword=keyword,
        max_num=amount)

# download room images
def gather_dataset():
    crawl("kitchen", 2000, folder="dataset/raw_data/kitchen_room/")
    crawl("livingroom", 2000, folder="dataset/raw_data/living_room/")
    crawl("bathroom", 2000, folder="dataset/raw_data/bath_room/")
    crawl("bedroom", 2000, folder="dataset/raw_data/bed_room/")


if __name__ == "__main__":
    gather_dataset()
