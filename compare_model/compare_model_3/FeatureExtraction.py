#! /usr/bin/env python
# coding:utf-8

import math
import io

class Util_Feature_Extraction:
	E_p_base = 1.0
	def __CalculateInformationGain(self,train):
		word = {}
		lastword = {}
		docu_num = 0
		docu_1_num =0
		docu_0_num = 0

		#print(word)

		for line in train:
			docu_num +=1
			length = len(line)
			if line[0] == '1':

				for col in line[1:length]:
					docu_1_num += 1
					if col in word:
						word[col][0] +=1
					else:
						word[col] = [self.E_p_base+1,self.E_p_base,self.E_p_base,self.E_p_base]
			elif line[0] == '0':

				for col in line[1:length]:
					docu_0_num += 1
					if col in word:
							word[col][2] +=1
					else :
						word[col] = [self.E_p_base,self.E_p_base,self.E_p_base+1,self.E_p_base]
			else:
				print(line),
				print("error")


			#print(docu_num)
		N1 = 0.0
		N0 = 0.0
		for line in train:
			if line[0] == '0':
				N0 +=1
			else:
				N1 +=1
		#del train
		if docu_num != N1+N0:
			print("error")
			exit()
		total_word = 0
		for w in word:
			total_word +=1
			word[w][1] = docu_1_num - word[w][0]
			word[w][3] = docu_0_num - word[w][2]
		#for w in word:
			#if docu_num != word[w][0]+word[w][1]+word[w][2]+word[w][3]:
				#print("error")
				#exit()
		N1 = docu_1_num
		N2 = docu_0_num
		E_s = -((N1/(N1+N0))*math.log(N1/(N1+N0)) + (N0/(N1+N0))*math.log(N0/(N1+N0)))# get the system etropy
		#print(E_s)
##############################################
		info_gain={}
		num = 0
		for w in word:
			num +=1
			A = word[w][0]#
			B = word[w][2]#
			C = word[w][1]#
			D = word[w][3]#
			#print(A,B,C,D)
			if A+B+C+D != docu_1_num +docu_0_num :
				print("Number Not Equal ---> error")
				exit()
			A_part = (A/(A+B)) * math.log(A/(A+B),2)

			B_part = (B/(A+B)) * math.log(B/(A+B),2)

			C_part = (C/(C+D)) * math.log(C/(C+D),2)

			D_part = (D/(C+D)) * math.log(D/(C+D),2)
			info_gain.setdefault(w,E_s +((A+B)/(N1+N0))*(A_part + B_part) + ((C+D)/(N1+N0))*(C_part + D_part))
			if(info_gain[w] == 0):
				print("Info Gain equal to 0 ---> error")
				exit()
			#print("processing is "+str(float(num)/float(total_word)))

		del word

		####################################################################3
		info_gain_tmp= sorted(info_gain.iteritems(), key=lambda d:d[1], reverse = True)
		# for k in info_gain_tmp:
		# 	print(k[0])
			#print(info_gain[k])
		return info_gain_tmp

	def __OutPutFeature(self,info_gain_sorted,Num,OutputFeature,OutputAllFeature):
		result_file = open(OutputFeature,"w") #

		for k in info_gain_sorted[0:Num]:
			result_file.write(str(str(k).encode('utf-8')).strip())
			result_file.write('\r\n')
		result_file.flush()
		result_file.close()

		all_result_file = open(OutputAllFeature,"w") #

		for k in info_gain_sorted:
			#result = str(k)
			all_result_file.write(str(str(k).encode('utf-8')).strip())
			all_result_file.write('\r\n')
		all_result_file.flush()
		all_result_file.close()


	def GetFeature(self,train,NumOfFeature,outputfeaturefile,outputallfeaturefile):
		info_gain = self.__CalculateInformationGain(train)
		#self.__OutPutFeature(info_gain,NumOfFeature,outputfeaturefile,outputallfeaturefile)
		feature = []
		for k in info_gain[0:NumOfFeature]:
			#print(k)
			# s = str(k[0]).split(',')
			# s[0] = s[0][2:len(s[0])-1]
			# s[0] = s[0].strip()
			feature.append(k[0])
		return feature
