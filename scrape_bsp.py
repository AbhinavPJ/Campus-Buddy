from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd

# Setup browser
options = Options()
# Comment this during debugging
# options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=options)
driver.get("https://www.bspiitd.com/humans-of-iitd")

# Wait until post <a> tags load
WebDriverWait(driver, 15).until(
    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a.O16KGI'))
)

# Parse page
soup = BeautifulSoup(driver.page_source, "html.parser")
post_links = soup.select('a.O16KGI')

posts = []
for a in post_links:
    href = a.get("href")
    title_tag = a.find("h2")
    title = title_tag.text.strip() if title_tag else "No Title"
    posts.append({"Title": title, "URL": href})

driver.quit()

# Save to CSV
df = pd.DataFrame(posts)
df.to_csv("bsp_editor.csv", index=False)
print(f"✅ Extracted {len(df)} article links to bsp_iitd_intern_links.csv")
