import numpy as np
import preprocessor

class DecisionTreeRunner(object):
	def __init__(self, data, clf):
		training_data = data["training"]
		testing_data =  data["test"]
		self.train  = preprocessor.normalize(training_data[:, :-1])
		self.train_labels = training_data[:, -1:].reshape(training_data.shape[0],)
		print(self.train_labels.shape)
		
		self.test = preprocessor.normalize(testing_data[:, :-1])
		self.test_labels = testing_data[:, -1:]
		self.confusion_matrix = np.zeros((2, 2))
		self.clf = clf
		self.accuracy = 0
		self.results = dict()
		self.failed_predictions = dict()
		
	def run(self):
		wrong_prediction_count = 0
		self.clf.fit(self.train, self.train_labels)
		for i in range(self.test_labels.shape[0]):
			test_data = self.test[i].reshape(1, -1)  #TODO: reshape all data set instead of individual
			predicted = int(self.clf.predict(test_data))
			# log_prob = int(self.clf.predict_log_proba(test_data[i]))
			actual = int(self.test_labels[i])
			self.confusion_matrix[predicted][actual] += 1
			self.results[i] = {
				'predicted_survival': predicted,
				'actual': actual
			}
			# 'predicted_probability_survival': log_prob, #TODO: fix this
			
			if predicted != actual:
				self.failed_predictions[wrong_prediction_count] = {
					'predicted_survival': predicted,
					'actual': actual,
					'data': test_data
				}
				wrong_prediction_count += 1
		self.wrong_prediction_count = wrong_prediction_count
		self.accuracy = np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)
			
	def graph_results(self):
		print("PlaceHolder") #TODO: implement
		
	def score(self):
		return self.clf.score(self.test, self.test_labels) #TODO: test
	
	def feature_importance(self):
		return self.clf.feature_importances_
	
	def print_accuracy(self):
		print(self.accuracy)
