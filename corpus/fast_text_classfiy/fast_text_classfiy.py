"""
this mode for preparing data which fasttext need
"""
from tqdm import tqdm, trange
from config import config
from utils.cut_word import Cut
import json
class process_classfiy(object):

    def __init__(self):

        self.cut = Cut()
        self.count_QA = 0
        self.count_Chat = 0
        self.classfiy_save = open(config.process_save_classfiy.get("classfiy"),'a+',encoding='utf-8')
        self.xiaohuangji_save = open(config.process_save_classfiy.get("xiaohuangji"),'a+',encoding='utf-8')
        self.QA_save = open(config.process_save_classfiy.get("QA"),'a+',encoding='utf-8')

    def process_xiaohuangji(self):
        flag = 0
        for line in tqdm(
                open(config.data_path.get("xiaohuangji"),'r',encoding='UTF-8').readlines(),
            desc="process_xiaohuangji"
        ):
            if (line.startswith("E")):
                flag = 0
                continue
            elif(line.startswith("M")):
                if(flag==0):
                    line = line[1:].strip()
                    flag = 1
                else:
                    continue
            line_cuted = " ".join(self.cut.cut(line))+"\t"+"__label__chat"
            self.xiaohuangji_save.write(line_cuted+"\n")
            self.classfiy_save.write(line_cuted+"\n")
            self.count_Chat+=1
        self.xiaohuangji_save.close()

    def process_qa(self):
        """
        this is for qa processing
        :return:
        """

        for line in tqdm(
            open(config.data_path.get("QA"), 'r', encoding='utf8'),
            desc="process_qa"
        ):
            data_line = json.loads(line)
            line_cuted = self.cut.cut(data_line.get("Q"))
            line_cuted = " ".join(line_cuted)+"\t"+"__label__QA"
            self.QA_save.write(line_cuted+"\n")
            self.classfiy_save.write(line_cuted + "\n")
            self.count_QA+=1
        self.QA_save.close()

    def process(self):
        #load xiaohuangji
        self.process_xiaohuangji()
        #load qa
        self.process_qa()
        self.classfiy_save.close()
        print("\033[0;32;40m all processing is finished in classfiy!\033[0m")
        print("All data is:",self.count_QA+self.count_Chat,
              "\n The Chat numbers is:",self.count_Chat,
              "\n The QA numbers is:",self.count_QA
              )

if __name__ == '__main__':
    process_classfiy = process_classfiy()
    process_classfiy.process()