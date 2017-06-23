import requests
import re

def get_ip_info(ip):
    url = 'http://freegeoip.net/json/' + ip
    r = requests.get(url)
    js = r.json()
    return js

def clean_output(out):
    out = out.strip()
    
    # clear parenthesis
    out = out.replace("(", "")
    out = out.replace(")", "")
    
    # clear triple or more new lines
    newline_regex = re.compile('\n\n*\n')
    out = newline_regex.sub('\n\n', out)
    
    # remove the incomplete last line
    out = "\n".join(out.split('\n')[:-1])
    
    out = out.strip() # in case the last line is alone
    
    return out + " ..."
    
def clean_input(prime):
    prime = prime.strip()
    prime = prime.capitalize()
    return prime + " "