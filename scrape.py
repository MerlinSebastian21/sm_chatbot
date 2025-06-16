# scrape.py
import os
import requests
from bs4 import BeautifulSoup
import asyncio, json
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer

 # SCRAPE SITEMAP & COLLECT CLEAN PAGE LINKS
def return_xml(url):
    response = requests.get(url)
    #response.raise_for_status()
    return BeautifulSoup(response.content, "xml")

def get_all_page_links(sitemap_index_url):
    page_links = []
    index_xml = return_xml(sitemap_index_url)
    sitemap_urls = [loc.text for loc in index_xml.find_all("loc")]

    for sitemap_url in sitemap_urls:
        print(f"Processing: {sitemap_url}")
        sitemap_xml = return_xml(sitemap_url)
        links = [loc.text for loc in sitemap_xml.find_all("loc")]
        clean_links = [
            link for link in links if not any(link.lower().endswith(ext) for ext in [
                ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
                ".pdf", ".docx", ".mp4", ".js", ".css"
            ])
        ]
        page_links.extend(clean_links)

    return list(set(page_links))

sitemap_index = "https://www.simelabs.com/sitemap_index.xml"
urls = get_all_page_links(sitemap_index)

print(f"\n Total clean page URLs collected: {len(urls)}")

async def load_and_clean(urls):
    print("\n Loading web pages asynchronously...")
    loader = AsyncHtmlLoader(urls)
    docs =  loader.load()

    print(f"Loaded {len(docs)} raw HTML documents.")

    print(" Cleaning HTML to plain text...")
    transformer = Html2TextTransformer()
    docs_clean = transformer.transform_documents(docs)

    print(f" Cleaned {len(docs_clean)} documents.")
    return docs_clean

docs_clean = asyncio.run(load_and_clean(urls))
with open("clean_docs.json", "w") as f:
        json.dump([doc.dict() for doc in docs_clean], f)
print(f"Saved {len(docs_clean)} cleaned docs to clean_docs.json")