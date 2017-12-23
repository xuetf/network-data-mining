#! /usr/bin/env python
# coding:utf-8

import jieba
import re

class Util_WordSegmentation:

	def CutList(self,list):
	#	out_file = open('output_word_file.txt','w')
		Result = []
		for line in list:
			#print(line)
			seg_list = jieba.cut(line.strip(), cut_all=True )#
			result = ''
		#	result ="|".join(seg_list)
			for line in seg_list:
				result += line.encode('utf-8')
				result += '|'
			#print(result)
			result = re.sub('xx*','|',result)
			result = re.sub('\|\|*','|',result)
			#result = result[0:len(result)-1]
			result = result.strip()
			if result != '' and result[0] == '|':
				result = result[1:len(result)]
			result = result.strip()
			#print(result)
			#out_file.write(result+'\r\n')
			Result.append(result)
		# for line in Result:
		# 	print(line)
		# print(Result)
		#out_file.flush()
		#out_file.close()
		print("Word Segmentation is down!!!\n")
		return Result

# list = ['就开始放松空间能否快速减肥开始减肥','技术的积分沙发是减肥的话']
# w_Msg = Util_WordSegmentation()
#
# w_Msg.CutList(list)
