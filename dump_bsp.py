import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

# Load Title + URL CSV
df = pd.read_csv("bsp_editor.csv")  # change to your actual filename if needed

# Setup Selenium
options = Options()
# comment this out to see the browser
options.add_argument("--headless")
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=options)

# Output file
output_path = "bsp_all_articles.txt"
with open(output_path, "a", encoding="utf-8") as f:
    for index, row in df.iterrows():
        title = row['Title']
        url = row['URL']
        print(f"🔗 [{index+1}/{len(df)}] {title}")
        
        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(1)  # Let JS finish rendering

            soup = BeautifulSoup(driver.page_source, "html.parser")
            article_text = soup.body.get_text(separator="\n", strip=True)

            f.write(f"\n{'='*80}\n")
            f.write(f"📝 {title}\n📍 {url}\n")
            f.write(f"{'-'*80}\n")
            f.write(article_text)
            f.write("\n")
        except Exception as e:
            print(f"⚠️ Error at {url}: {e}")
            continue

driver.quit()
print(f"\n✅ Dumped all articles to {output_path}")
