import numpy as np
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def readList(list_path,ignore_head=False, sort=True):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    if sort:
        lists.sort(key=natural_keys)
    return lists

def copy_txt_file(source_file, destination_file):
    # 打开源文件和目标文件
    with open(source_file, "r", encoding="utf-8") as source:
        with open(destination_file, "w", encoding="utf-8") as destination:
            # 读取源文件的内容
            content = source.read()
            # 将源文件的内容写入目标文件
            destination.write(content)
            
def write_to_txt(dir,obj):  #先清空内容再写入内容

    with open(dir,"w",encoding='utf-8') as file:
        file.writelines(obj)
        file.close()
            
