import threading
import time
import scrapy
from scrapy.crawler import CrawlerRunner
from twisted.internet import reactor
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ------------------- متغيرات الذاكرة -------------------
crawled_data = []  # تخزين البيانات في الذاكرة بدلاً من الملفات

# ------------------- إعدادات الزحف على الويب -------------------

class WebCrawler(scrapy.Spider):
    name = "web_crawler"
    start_urls = ['https://colle-pedia.blogspot.com']  # ضع الرابط الذي تريد الزحف إليه هنا

    def parse(self, response):
        post_body = response.css('div.post-body').get()
        if post_body:
            crawled_data.append(post_body)

        for next_page in response.css('a::attr(href)'):
            yield response.follow(next_page.get(), self.parse)

# ------------------- دالة الزحف المستمر -------------------

def start_crawling():
    while True:
        try:
            runner = CrawlerRunner()
            deferred = runner.crawl(WebCrawler)
            deferred.addBoth(lambda _: reactor.stop())
            reactor.run(installSignalHandlers=False)
        except Exception as e:
            print(f"خطأ أثناء الزحف: {e}")
        time.sleep(3600)  # انتظر ساعة واحدة قبل بدء عملية الزحف من جديد

# ------------------- إعدادات معالجة البيانات وتدريب النموذج -------------------

def train_model():
    global model, tokenizer  # لجعل النموذج والـ tokenizer متاحين للاستخدام في الواجهة API

    while True:
        try:
            if crawled_data:
                # تنظيف البيانات وتحضيرها للتدريب
                data = pd.DataFrame(crawled_data, columns=['post_body'])
                data['clean_text'] = data['post_body'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

                # تحميل النموذج والـ tokenizer
                model = GPT2LMHeadModel.from_pretrained('gpt2')
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

                # تحويل النصوص إلى صيغة إدخال للنموذج
                train_encodings = tokenizer(data['clean_text'].dropna().tolist(), truncation=True, padding=True, return_tensors='pt')

                # إعدادات التدريب
                training_args = TrainingArguments(
                    output_dir='./results',
                    num_train_epochs=1,
                    per_device_train_batch_size=4,
                    logging_dir='./logs',
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_encodings
                )

                trainer.train()
        except Exception as e:
            print(f"خطأ أثناء التدريب: {e}")

        time.sleep(7200)  # انتظر ساعتين قبل بدء عملية التدريب من جديد

# ------------------- إعداد FastAPI -------------------

app = FastAPI()

class RequestData(BaseModel):
    text: str

@app.post("/generate-response")
def generate_response(request: RequestData):
    inputs = tokenizer(request.text, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=150)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response_text}

# ------------------- تشغيل FastAPI وعمليات الخلفية -------------------

if __name__ == "__main__":
    # إنشاء خيوط (threads) للزحف والتدريب لتعمل بشكل متزامن مع واجهة FastAPI
    crawler_thread = threading.Thread(target=start_crawling)
    trainer_thread = threading.Thread(target=train_model)

    # بدء الخيوط
    crawler_thread.start()
    trainer_thread.start()

    # بدء تشغيل واجهة FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)
