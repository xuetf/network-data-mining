#! /usr/bin/env python
# coding:utf-8

import math
import WordSegmentation
import FeatureExtraction
import io
from sklearn.metrics import *
import csv

class Util_Classify_Message:

	InputTrainingFile = "training_result.txt"
	OutPutResultFile = "testing_classify_result.txt"
	ModelFile = 'Model.csv'
	# InputTestingFile = "testing_result.txt"
	InputOriginalTestingFile = "testing.txt"
	OutputFeature = "feature.txt"
	OutputAllFeature = "all_feature.txt"

	NumOfFeature = 200#
	E_p_base = 0.00000001#0.001
	Py_0 = 0.0
	Py_1 = 0.0
	Pfeature = {}

	def __init__(self,Training_file):
		self.InputTrainingFile = Training_file

	def GetTraingData(self):
		training_file = io.open(self.InputTrainingFile,'r',encoding = 'utf-8')
		train = []
		for line in training_file:
			train .append(line)
		w_seg = WordSegmentation.Util_WordSegmentation()
		cut_train = w_seg.CutList(train)
		tmp_train = []
		for line in cut_train:
			tmp_train.append(line.split('|'))
			#print(line)
		training_file.close()
		# for line in tmp_train:
		# 	for line2 in line:
		# 		print(line2),
		# 		print(' '),
		# 	print('')
		return tmp_train
	def GetTestingData(self,InputTestingFileName):
		testing_file = io.open(InputTestingFileName,'r',encoding = 'utf-8')
		test_result = []
		for line in testing_file:
			test_result .append(line)
		w_seg = WordSegmentation.Util_WordSegmentation()
		cut_test = w_seg.CutList(test_result)
		tmp_test = []
		for line in cut_test:
			tmp_test.append(line.split('|'))
			#print(line)
		testing_file.close()

		return tmp_test

	def Training(self,fold):
		print("Evaluate the model by "+str(fold)+"-fold(it will cost some time to training the model)......\n")
		util_feature = FeatureExtraction.Util_Feature_Extraction()

		train = self.GetTraingData()
		accuracy = 0.0
		percision = [0.0,0.0]
		recall = [0.0,0.0]
		F1 = [0.0,0.0]
		for i in range(fold):
			Py_0 = 0.0
			Py_1 = 0.0
			Pfeature = {}
			lastPfeature = {}
			tmp_train =[]
			#print(len(train))
			for line in train[0:int((i)*len(train)/fold)]:#
				tmp_train.append(line)

			for line in train[int((i+1)*len(train)/fold):int((fold)*len(train)/fold)]:#
				tmp_train.append(line)

			tmp_feature = util_feature.GetFeature(tmp_train,self.NumOfFeature,self.OutputFeature,self.OutputAllFeature)

			# + -
			for f in tmp_feature:
				Py_0 += self.E_p_base
				Py_1 += self.E_p_base
				Pfeature[f] = [self.E_p_base,self.E_p_base]

			for line in tmp_train:
				if line[0] == '0':

					for col in line[1:len(line)]:
						if col in Pfeature :
							Py_0 +=1
							Pfeature[col][1] += 1
				if line[0] == '1':

					for col in line[1:len(line)]:

						if col in Pfeature:
							Py_1 +=1
							Pfeature[col][0] += 1

			#for f in Pfeature:
				#if(Pfeature[f][0]+Pfeature[f][1] != Py_0+Py_1)
			#print(Py_1+Py_0)
			TP = 0.0
			FP = 0.0
			TN = 0.0
			FN = 0.0
			print(str(i+1)+"th training is down! \n")

			Evaluation_Result = []
			Result = []
			for line in train[int((i)*len(train)/fold):int((i+1)*len(train)/fold)]:#
				current_P1 = math.log((Py_1/(Py_0+Py_1)),10)
				current_P0 = math.log((Py_0/(Py_0+Py_1)),10)
				for f in Pfeature:
					if f in line[1:len(line)]:
						current_P1 += math.log(1.25 * Pfeature[f][0]/Py_1,10)
						current_P0 += math.log(2 * Pfeature[f][1]/Py_0,10)
				#for col in line[1:len(line)-1]:

					#if col in Pfeature:
						#current_P1 *= Pfeature[col][0]/Py_1
						#current_P0 *= Pfeature[col][1]/Py_0
				if(current_P1 == 0 or current_P0 ==0):
					print("equal to 0 --->error")
					exit()
				if current_P1 >= current_P0:
					Evaluation_Result.append(1)
				else:
					Evaluation_Result.append(0)
				if line[0] == '1':
					Result.append(1)
				elif line[0] == '0':
					Result.append(0)
				# '''
				# if (current_P1 >= current_P0 and line[0] == '1'):
				# 	TN +=1
				# elif (current_P1 < current_P0 and line[0] == '0'):
				# 	TP += 1
				# elif (current_P1 >= current_P0 and line[0] == '0'):
				# 	FN += 1
				# elif (current_P1 < current_P0 and line[0] == '1'):
				# 	FP += 1
				# else:
				# 	print("error result")
				# 	'''
				# '''
				# if (current_P1 >= current_P0 and line[0] == '1'):
				# 	TP +=1
				# elif (current_P1 < current_P0 and line[0] == '0'):
				# 	TN += 1
				# elif (current_P1 >= current_P0 and line[0] == '0'):
				# 	FP += 1
				# elif (current_P1 < current_P0 and line[0] == '1'):
				# 	FN += 1
				# else:
				# 	print("error result")
				# '''
			#print("Right is"+ str((TP+TN)/(TP+TN+FP+FN)))
			Evaluation = precision_recall_fscore_support(Result,Evaluation_Result,pos_label = 1)
			Accuracy = accuracy_score(Result,Evaluation_Result)
			#print(Evaluation)
			# print("Percision OF "+str(i+1)+"th training is " + str(TP/(TP+FP)))
			# print("Recall OF "+str(i+1)+"th training is " + str(TP/(TP+FN)))
			# print("F1 OF "+str(i+1)+"th training is " + str(2*TP/(2*TP+FP+FN)) + '\n\n')
			print("Accuracy OF "+str(i+1)+"th training is "+str(Accuracy))
			print("Percision OF "+str(i+1)+"th training is " + str(Evaluation[0]))
			print("Recall OF "+str(i+1)+"th training is " + str(Evaluation[1]))
			print("F1 OF "+str(i+1)+"th training is " + str(Evaluation[2])+ '\n\n')
			percision += Evaluation[0]
			recall += Evaluation[1]
			F1 += Evaluation[2]
			accuracy += Accuracy
			#print("by tool")
		print("Now Get The Final Evaluation Of This Model:")
		print("Accuracy is " + str(accuracy/5)+"\n")
		print("percision is " + str(percision/5))
		print("recall is "+ str(recall/5))
		print("F1 is " + str(F1/5)+"\n")

	def StoreModel(self,Py_0,Py_1,Pfeature):
		csvFile = open(self.ModelFile,'w')
		writer = csv.writer(csvFile)
		writer.writerow([Py_0,Py_1])
		for key in Pfeature:
			writer.writerow([key, Pfeature[key]])
		csvFile.close()
	def LoadModel(self):
		print("Now, It is starting loding the model")
		csvFile = open(self.ModelFile,'r')
		reader = csv.reader(csvFile)
		result = []
		for item in reader:
			result.append(item)
		self.Py_0 = float(result[0][0])
		self.Py_1 = float(result[0][1])
		for line in result[1:len(result)]:
			string = line[1][1:len(line[1])-1]
			feature_weight = [0.0,0.0]
			feature_weight[0] = float(string.split(',')[0])
			feature_weight[1] = float(string.split(',')[1])
		#	print(string)
			self.Pfeature.setdefault(line[0],feature_weight)
		# for key in self.Pfeature:
		# 	print(self.Pfeature[key])
		print("Loding Model is Down!!!")
	def Train_Model(self):
		print("Now, It is training all of this set!")
		util_feature = FeatureExtraction.Util_Feature_Extraction()

		train = self.GetTraingData()
		Py_0 = 0.0
		Py_1 = 0.0
		Pfeature = {}
		lastPfeature = {}
		tmp_train =[]

		for line in train:
			tmp_train.append(line)

		tmp_feature = util_feature.GetFeature(tmp_train,self.NumOfFeature,self.OutputFeature,self.OutputAllFeature)

		# + -
		for f in tmp_feature:
			Py_0 += self.E_p_base
			Py_1 += self.E_p_base
			Pfeature[f] = [self.E_p_base,self.E_p_base]

		for line in tmp_train:
			if line[0] == '0':

				for col in line[1:len(line)]:

					if col in Pfeature :
						Py_0 +=1
						Pfeature[col][1] += 1
			if line[0] == '1':

				for col in line[1:len(line)]:

					if col in Pfeature:
						Py_1 +=1
						Pfeature[col][0] += 1
		self.StoreModel(Py_0,Py_1,Pfeature)
		print("Training all the set is down, The model is kept in the "+ self.ModelFile)

	def GetResult(self,InputTestingFileName):
		test = self.GetTestingData(InputTestingFileName)
		Py_1 = self.Py_1
		Py_0 = self.Py_0
		Pfeature = self.Pfeature
		for line in test:
			current_P1 = math.log(Py_1/(Py_0+Py_1),10)
			current_P0 = math.log(Py_0/(Py_0+Py_1),10)

			for f in Pfeature:
				if f in line[1:len(line)]:
					current_P1 += math.log(1.25*Pfeature[f][0]/Py_1,10)
					current_P0 += math.log(2*Pfeature[f][1]/Py_0,10)
			#for col in line[1:len(line)-1]:
				#if col in Pfeature:
					#current_P1 *= Pfeature[col][0]/Py_1
					#current_P0 *= Pfeature[col][1]/Py_0
			if (current_P1 >= current_P0 ):
				line.insert(0,'1 ')
			elif (current_P1 < current_P0):
				line.insert(0,'0 ')
		print("calculating data is down!\nThen write it to the file --->"+self.OutPutResultFile)

		orignalTestingFile = open(self.InputOriginalTestingFile)
		orignalTest = []
		for line in orignalTestingFile:
			orignalTest.append(line)
		orignalTestingFile.close()
		all_result_file = open(self.OutPutResultFile,"w")
		for i in range(len(test)):
			all_result_file.write(test[i][0]+' '+orignalTest[i]+'\r\n')

		print('Now Writing file is down!')
		all_result_file.flush()
		all_result_file.close()


	def Predict(self,testing):
		Py_1 = self.Py_1
		Py_0 = self.Py_0
		Pfeature = self.Pfeature
		Result = []
		w_seg = WordSegmentation.Util_WordSegmentation()
		test = w_seg.CutList(testing)
		for line in test:
			current_P1 = math.log(Py_1/(Py_0+Py_1),10)
			current_P0 = math.log(Py_0/(Py_0+Py_1),10)
			for f in Pfeature:
				if f in line[1:len(line)]:
					current_P1 += math.log(1.25*Pfeature[f][0]/Py_1,10)
					current_P0 += math.log(2*Pfeature[f][1]/Py_0,10)
			#for col in line[1:len(line)-1]:
				#if col in Pfeature:
					#current_P1 *= Pfeature[col][0]/Py_1
					#current_P0 *= Pfeature[col][1]/Py_0
			if (current_P1 >= current_P0 ):
				Result.append(1)
			elif (current_P1 < current_P0):
				Result.append(0)
		return Result

if __name__ == '__main__':
	InputTrainingFileName = 'training.txt'
	InputTestingFileName = 'testing.txt'
	c_Msg = Util_Classify_Message(InputTrainingFileName)
	#c_Msg.Training(5)

	#c_Msg.Train_Model()
	c_Msg.LoadModel()
	#c_Msg.GetResult(InputTestingFileName)
	test = ['x强度等级水泥的必要性和可行性进行深入研究','.x月xx日推出凭证式国债x年期x.xx.xx%，x年期x.xx%到期一次还本付息。真情邮政，为您竭诚服务！  咨询电话xxxx-xx','庆xx节本会所优惠活动，为答谢新老顾客的支持与厚爱，，面部特卡:xxx元/xx次，身体活动，带脉减小肚腩:xxxx元/xx次，，肠胃','','']
	result = c_Msg.Predict(test)
	print(result)
