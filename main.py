from flask import Flask, request, jsonify
from transformers import pipeline
import scrapy
from scrapy.crawler import CrawlerProcess
from typing import List

app = Flask(__name__)

# استخدام نموذج pre-trained من مكتبة Transformers
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# كلاس الزحف باستخدام Scrapy
class MySpider(scrapy.Spider):
    name = "simple_spider"
    start_urls = []  # سنقوم بتحديث هذه القائمة بالرابط المطلوب للزحف
    extracted_data = []

    def parse(self, response):
        paragraphs = response.css('p::text').getall()
        self.extracted_data.extend(paragraphs)

# دالة الزحف وجلب البيانات
def crawl_website(url: str) -> List[str]:
    process = CrawlerProcess()
    MySpider.start_urls = [url]
    process.crawl(MySpider)
    process.start()
    return MySpider.extracted_data

# عرض النتائج في المتصفح مباشرة
@app.route("/", methods=["GET"])
def read_root():
    url = request.args.get("url", default="https://example.com")
    data = crawl_website(url)
    return jsonify({"Extracted Data": data})

# سؤال وإجابة مباشرة من المتصفح
@app.route("/ask", methods=["GET"])
def ask_question():
    question = request.args.get("question")
    context = request.args.get("context")
    result = qa_pipeline(question=question, context=context)
    return jsonify({"Answer": result['answer']})

if __name__ == "__main__":
    app.run()
