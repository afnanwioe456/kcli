from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep

PATH = 'E:\\Python_projects\\chrome_dev\\chrome_dev120.0.6099.109'
AVAILABLE = False

# 配置选项
options = webdriver.ChromeOptions()
options.add_experimental_option('debuggerAddress', '127.0.0.1:9999')
options.add_argument('incognito')
options.add_argument('disable-infobars')
options.add_argument('disable-web-security')  # 关闭安全策略
options.add_argument('allow-running-insecure-content')  # 允许运行不安全的内容
options.add_argument('ignore-certificate-errors')  # 忽略证书错误
s = Service(PATH + '\\chromedriver.exe')


def natural_command(text):
    if not AVAILABLE:
        return
    args = text.split()
    args.append('.')
    content = ''.join(args[1:])
    with webdriver.Chrome(service=s, options=options) as driver:
        print(len(driver.window_handles))
        driver.switch_to.window(driver.window_handles[0])
        # 为什么是0？？？手动打开的新标签页为0
        rewrite_button = driver.find_element(By.XPATH, '/html/body/div[1]/div[1]/div[2]/main/div[2]/div['
                                                       '1]/div/div/div/div[10]/div/div/div[2]/div[2]/div[2]/div[2]')
        rewrite_button.click()
        js_str = (f'''var d = document.evaluate(
                  "/html/body/div[1]/div[1]/div[2]/main/div[2]/div[1]/div/div/div/div[10]/div/div/div[2]/div[2]/div/textarea",
                  document,
                  null,
                  XPathResult.FIRST_ORDERED_NODE_TYPE,
                  null
                  ).singleNodeValue;d.value="{content}"
                  ''')
        driver.execute_script(js_str)
        input_box = driver.find_element(By.XPATH, '/html/body/div[1]/div[1]/div[2]/main/div[2]/div[1]/div/div/div/div['
                                                  '10]/div/div/div[2]/div[2]/div/textarea')
        input_box.click()
        input_box.send_keys(Keys.BACK_SPACE)
        submit_button = driver.find_element(By.XPATH, '/html/body/div[1]/div[1]/div[2]/main/div[2]/div['
                                                      '1]/div/div/div/div[10]/div/div/div[2]/div[2]/div/div/button[1]')
        submit_button.click()
        sleep(5)
        result_code = driver.find_element(By.XPATH,
                                          '/html/body/div[1]/div[1]/div[2]/main/div[2]/div[1]/div/div/div/div['
                                          '11]/div/div/div[2]/div[2]/div[1]/div/div/pre/div/div[2]/code')
        result = result_code.text
        print(result)
        return result


if __name__ == '__main__':
    natural_command('!n 以高优先级发射中继卫星')
