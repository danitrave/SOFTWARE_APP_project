import unittest, os, sys, itertools
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from project import DataSetReader, StatisticsRecorder, TrainingGraph, TestGraph, ValidationGraph, Classification, MachineLearning, ConfusionMatrixCreator
from io import StringIO
from SPARQLWrapper import SPARQLWrapper, JSON
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import output_template
    
#I test case.        
class DataSetReaderTestCase(unittest.TestCase): 
    def setUp(self):
        self.dataSetReader = DataSetReader
        self._testSPARQL = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT (COUNT(DISTINCT ?split) AS ?inds)
            FROM <http://wit.istc.cnr.it/geno/training>
            WHERE {
                ?split <http://wit.istc.cnr.it/geno/sequence> ?sequence
            }
        """

    def tearDown(self):
        self.dataSetReader = None

    def testQuery(self):
        sparql = self.dataSetReader.query(self._testSPARQL)
        self.assertEqual(sparql, {'head': {'link': [], 'vars': ['inds']}, 'results': {'distinct': False, 'ordered': True, 'bindings': [{'inds': {'type': 'typed-literal', 'datatype': 'http://www.w3.org/2001/XMLSchema#integer', 'value': '1829'}}]}}, 'The DataSetReader.query output is incorrect.')

    def testQueryTypeCheck(self):
        sparql = self.dataSetReader.query(self._testSPARQL)
        self.assertEqual(type(sparql), dict, 'The type of DataSetReader.query output is incorrect.')

    def testQueryKeys(self):
        sparql = self.dataSetReader.query(self._testSPARQL)
        self.assertEqual(list(sparql.keys()), ['head', 'results'], 'The DataSetReader.query resulting dictionary keys are incorrect.')

    def testQueryItems(self):
        sparql = self.dataSetReader.query(self._testSPARQL)
        self.assertEqual(list(sparql.items()), [('head', {'link': [], 'vars': ['inds']}), ('results', {'distinct': False, 'ordered': True, 'bindings': [{'inds': {'type': 'typed-literal', 'datatype': 'http://www.w3.org/2001/XMLSchema#integer', 'value': '1829'}}]})], 'The DataSetReader.query resulting dictionary items are incorrect.')

    def testQueryKeysLen(self):
        sparql = self.dataSetReader.query(self._testSPARQL)
        self.assertEqual(len(list(sparql.keys())), 2, 'The number of keys in the DataSetReader.query resulting dictionary is incorrect.') #the data used in this test are static and decided by the team to test multiple times the output and avoid leaky data

    def testQueryItemsLen(self):
        sparql = self.dataSetReader.query(self._testSPARQL)
        self.assertEqual(len(list(sparql.items())), 2, 'The number of items in the DataSetReader.query resulting dictionary is incorrect.')

    def testQueryTrueKey(self):
        sparql = self.dataSetReader.query(self._testSPARQL)
        self.assertTrue('head' in list(sparql.keys()), 'The tested key is not in the DataSetReader.query resulting dictionary.')

    def testQueryFalseItem(self):
        sparql = self.dataSetReader.query(self._testSPARQL)
        self.assertFalse(('nofile') in list(sparql.items()), 'The tested item is in the DataSetReader.query resulting dictionary even if it shouldn''t.')
    
        
#II test case.
class StatisticsRecorderTestCase(unittest.TestCase):
    def setUp(self):
        self.statisticsRecorderTest = StatisticsRecorder

    def tearDown(self):
        self.statisticsRecorderTest = None

    def testStatistics(self):
        testStat = self.statisticsRecorderTest.computeStats()
        self.assertEqual(testStat, {'trainingGraphStats': '1829', 'testGraphStats': '1318', 'validationGraphStats': '30'}, 'The StatisticsRecorder.computeStats() output is incorrect.')

    def testStatisticsTypeCheck(self):
        testStat = self.statisticsRecorderTest.computeStats()
        self.assertEqual(type(testStat), dict, 'The type of StatisticsRecorder.computeStats() resulting output is incorrect.')

    def testStatisticsKeys(self):
        testStat = self.statisticsRecorderTest.computeStats()
        self.assertEqual(list(testStat.keys()), ['trainingGraphStats', 'testGraphStats', 'validationGraphStats'], 'The StatisticsRecorder.computeStats() resulting dictionary keys are incorrect.')

    def testStatisticsItems(self):
        testStat = self.statisticsRecorderTest.computeStats()
        self.assertEqual(list(testStat.items()), [('trainingGraphStats', '1829'), ('testGraphStats', '1318'), ('validationGraphStats', '30')], 'The StatisticsRecorder.computeStats() resulting dictionary items are incorrect.')

    def testStatisticsKeysLen(self):
        testStat = self.statisticsRecorderTest.computeStats()
        self.assertEqual(len(list(testStat.keys())), 3, 'The number of keys in the StatisticsRecorder.computeStats() resulting dictionary is incorrect.')
        
    def testStatisticsItemsLen(self):
        testStat = self.statisticsRecorderTest.computeStats()
        self.assertEqual(len(list(testStat.items())), 3, 'The number of items in the StatisticsRecorder.computeStats() resulting dictionary is incorrect.')

    def testStatisticsTrueKey(self):
        testStat = self.statisticsRecorderTest.computeStats()
        self.assertIn('trainingGraphStats', list(testStat.keys()), 'The tested key is not in the StatisticsRecorder.computeStats() resulting dictionary.')

    def testStatisticsFalseItem(self):
        testStat = self.statisticsRecorderTest.computeStats()
        self.assertFalse(('trainingGraphStats', '3000') in list(testStat.items()), 'The tested item is in the StatisticsRecorder.computeStats() resulting dictionary even if it shouldn''t.')

    def testStatisticsGreaterValue(self):
        testStat = self.statisticsRecorderTest.computeStats()
        self.assertGreater(int(list(testStat.values())[0]), 1800, 'The tested value in StatisticsRecorder.computeStats() resulting dictionary has resulted lower than expected.')
        
#III test case.
class TrainingGraphTestCase(unittest.TestCase):
    def setUp(self):
        self.TrainingGTest = TrainingGraph

    def tearDown(self):
        self.TrainingGTest = None

    def testTrainingGraph(self):
        testTG = self.TrainingGTest.getTrainingGraph()
        self.assertEqual(type(testTG),type(ValidationGraph.getValidationGraph()), 'The type of the training graph (TrainingGraph.getTrainingGraph()) is incorrect.') #conventionally, we use one of the other two graphs to check the type

    def testTrainingGraphList(self):
        testTG = self.TrainingGTest.getTrainingGraph()
        self.assertEqual(list(testTG), ['67', '67.1', '65', '71', '67.2', '84', '71.1', '67.3', '65.1', '84.1', '67.4', '65.2', '67.5', '65.3', '71.2', '71.3', '65.4', '71.4', '71.5', '67.6', '67.7', '65.5', '71.6', '67.8', '71.7', '65.6', '71.8', '67.9', '65.7', '71.9', '71.10', '84.2', '67.10', '84.3', '71.11', '84.4', '84.5', '67.11', '67.12', '65.8', '65.9', '71.12', '71.13', '71.14', '67.13', '67.14', '84.6', '84.7', '67.15', '71.15', '65.10', '71.16', '67.16', '67.17', '65.11', '71.17', '84.8', '67.18', '84.9', '71.18', 'EI'], 'The training graph doesn''t contain the expected values.')

    def testTrainingGraphListLen(self):
        testTG = self.TrainingGTest.getTrainingGraph()
        self.assertEqual(len(list(testTG)), 61, 'The number of elements present in the training graph is incorrect.')

    def testTrainingGraphItemPresent(self):
        testTG = self.TrainingGTest.getTrainingGraph()
        self.assertIn('67.2', list(testTG), 'The tested value is not in the training graph.')

    def testTrainingGraphListNotEmpty(self):
        testTG = self.TrainingGTest.getTrainingGraph()
        self.assertNotEqual(len(list(testTG)), 0, 'The training graph is empty.') 
        

#IV test case.
class TestGraphTestCase(unittest.TestCase):
    def setUp(self):
        self.TestGTest = TestGraph

    def tearDown(self):
        self.TestGTest = None

    def testTestGraph(self):
        testTSG = self.TestGTest.getTestGraph()
        self.assertEqual(type(testTSG),type(ValidationGraph.getValidationGraph()), 'The type of the test graph (TestGraph.getTestGraph()) is incorrect.')

    def testTestGraphList(self):
        testTSG = self.TestGTest.getTestGraph()
        self.assertEqual(list(testTSG), ['71', '67', '67.1', '65', '65.1', '71.1', '65.2', '65.3', '71.2', '65.4', '67.2', '71.3', '84', '84.1', '67.3', '67.4', '67.5', '84.2', '84.3', '71.4', '71.5', '84.4', '65.5', '65.6', '84.5', '65.7', '84.6', '67.6', '65.8', '71.6', '71.7', '84.7', '65.9', '65.10', '65.11', '84.8', '67.7', '67.8', '67.9', '65.12', '65.13', '84.9', '65.14', '65.15', '65.16', '84.10', '84.11', '67.10', '84.12', '67.11', '65.17', '71.8', '84.13', '65.18', '65.19', '65.20', '67.12', '84.14', '67.13', '84.15', 'EI'], 'The test graph doesn''t contain the expected values.')

    def testTestGraphListLen(self):
        testTSG = self.TestGTest.getTestGraph()
        self.assertEqual(len(list(testTSG)), 61, 'The number of elements present in the test graph is incorrect.')

    def testTestGraphItemPresent(self):
        testTSG = self.TestGTest.getTestGraph()
        self.assertIn('67.9', list(testTSG), 'The tested value is not in the test graph.')

    def testTestGraphListNotEmpty(self):
        testTSG = self.TestGTest.getTestGraph()
        self.assertNotEqual(len(list(testTSG)), 0, 'The test graph is empty.') 

#V test case.
class ValidationGraphTestCase(unittest.TestCase):
    def setUp(self):
        self.ValidationGTest = ValidationGraph

    def tearDown(self):
        self.ValidationGTest = None

    def testValidationGraph(self):
        testVG = self.ValidationGTest.getValidationGraph()
        self.assertEqual(type(testVG),type(TestGraph.getTestGraph()),'The type of the test graph (ValidationGraph.getValidationGraph() instance is incorrect.')

    def testValidationGraphListLen(self):
        testVG = self.ValidationGTest.getValidationGraph()
        self.assertEqual(len(list(testVG)), 1800, 'The number of elements present in the validation graph is incorrect.') #given the great amount of elements pressent in the validation graph, it is very computationally expensive to check their correctness, hence we avoided on purpose this test.

    def testValidationGraphItemPresent(self):
        testVG = self.ValidationGTest.getValidationGraph()
        self.assertIn('67.88', list(testVG), 'The tested value is not in the validation graph.')

    def testValidationGraphListNotEmpty(self):
        testVG = self.ValidationGTest.getValidationGraph()
        self.assertNotEqual(len(list(testVG)), 0, 'The validation graph is empty.')


#VI test case.
class MachineLearningTestCase(unittest.TestCase):
    def setUp(self):
        self._classifier = GaussianNB()
        self.MachineLearningTest = MachineLearning

    def tearDown(self):
        self._MachineLearningTest = None

    def testML(self):
        testMacLearn = self.MachineLearningTest.trainClassifier(self._classifier)
        self.assertEqual(testMacLearn, {'precision': 0.9261552467067371, 'recall': 0.8838268792710706, 'f1': 0.8946165412920902, 'confusion_matrix': 'static/confusion-matrix.png'}, 'The MachineLearning.TrainClassifier(classifier) output is incorrect.')

    def testMLTypeCheck(self):
        testMacLearn = self.MachineLearningTest.trainClassifier(self._classifier)
        self.assertEqual(type(testMacLearn), dict, 'The MachineLearning.TrainClassifier(classifier) type of the result is incorrect.')

    def testMLKeys(self):
        testMacLearn = self.MachineLearningTest.trainClassifier(self._classifier)
        self.assertEqual(list(testMacLearn.keys()), ['precision', 'recall', 'f1', 'confusion_matrix'], 'The keys of the MachineLearning.TrainClassifier(classifier) resulting dictionary are incorrect.')

    def testMLItems(self):
        testMacLearn = self.MachineLearningTest.trainClassifier(self._classifier)
        self.assertEqual(list(testMacLearn.items()), [('precision', 0.9261552467067371), ('recall', 0.8838268792710706), ('f1', 0.8946165412920902), ('confusion_matrix', 'static/confusion-matrix.png')], 'The items of the MachineLearning.TrainClassifier(classifier) resulting dictionary are incorrect.')

    def testMLKeysLen(self):
        testMacLearn = self.MachineLearningTest.trainClassifier(self._classifier)
        self.assertEqual(len(list(testMacLearn.keys())), 4, 'The number of the keys in the MachineLearning.TrainClassifier(classifier) resulting dictionary is incorrect.')
        
    def testMLItemsLen(self):
        testMacLearn = self.MachineLearningTest.trainClassifier(self._classifier)
        self.assertEqual(len(list(testMacLearn.items())), 4, 'The number of the items in the MachineLearning.TrainClassifier(classifier) resulting dictionary is incorrect.')

    def testMLTrueKey(self):
        testMacLearn = self.MachineLearningTest.trainClassifier(self._classifier)
        self.assertIn('f1', list(testMacLearn.keys()), 'The tested key is not in the MachineLearning.TrainClassifier(classifier) resulting dictionary.')

    def testMLFalseItem(self):
        testMacLearn = self.MachineLearningTest.trainClassifier(self._classifier)
        self.assertNotIn(('no file'), (testMacLearn.items()), 'The tested item has been found in the MachineLearning.TrainClassifier(classifier) resulting dictionary even if it shouldn''t.') 
                    
#VII test case.
class ClassificationTestCase(unittest.TestCase):
    def setUp(self):
        trainingSet = TrainingGraph.getTrainingGraph()
        X_train, y_train = trainingSet.iloc[:, :-1], trainingSet.iloc[:, -1]
        self._classifier = GaussianNB()
        self._classifier.fit(X_train, y_train)
        self.ClassificationTest = Classification

    def tearDown(self):
        self.ClassificationTest = None

    def testClassification(self):
        testCL = self.ClassificationTest.computeListIndividuals(self._classifier)
        self.assertEqual(testCL,['http://wit.istc.cnr.it/geno/ORAIGECA-904', 'http://wit.istc.cnr.it/geno/ORAHBG2F-269', 'http://wit.istc.cnr.it/geno/TARHBB-1909', 'http://wit.istc.cnr.it/geno/TARHBD-817', 'http://wit.istc.cnr.it/geno/ORAIGECA-1314', 'http://wit.istc.cnr.it/geno/ORAIGECA-378', 'http://wit.istc.cnr.it/geno/ORAHBG1F-614', 'http://wit.istc.cnr.it/geno/TARHBB-1560', 'http://wit.istc.cnr.it/geno/TARHBD-468', 'http://wit.istc.cnr.it/geno/ORAHBG2F-615', 'http://wit.istc.cnr.it/geno/ORAHBG2F-1492', 'http://wit.istc.cnr.it/geno/ORAIGECA-1393', 'http://wit.istc.cnr.it/geno/TARHBD-594', 'http://wit.istc.cnr.it/geno/ORAIGECA-583', 'http://wit.istc.cnr.it/geno/TARHBD-1884', 'http://wit.istc.cnr.it/geno/ORAHBG2F-392', 'http://wit.istc.cnr.it/geno/ORAHBG1F-391', 'http://wit.istc.cnr.it/geno/ORAIGECA-990', 'http://wit.istc.cnr.it/geno/TARHBB-1686', 'http://wit.istc.cnr.it/geno/TARHBB-2748', 'http://wit.istc.cnr.it/geno/ORAHBBE-2581', 'http://wit.istc.cnr.it/geno/ORAHBG2F-181', 'http://wit.istc.cnr.it/geno/ORAINVOL-2161', 'http://wit.istc.cnr.it/geno/ORARGIT-241', 'http://wit.istc.cnr.it/geno/ORAHBBPSE-6661', 'http://wit.istc.cnr.it/geno/ORAHBBPSE-2101', 'http://wit.istc.cnr.it/geno/ORAHBPSBD-2881', 'http://wit.istc.cnr.it/geno/ORAHBA01-121', 'http://wit.istc.cnr.it/geno/TARHBD-1981', 'http://wit.istc.cnr.it/geno/TARHBB-541'])

    def testClassificationTypeCheck(self):
        testCL = self.ClassificationTest.computeListIndividuals(self._classifier)
        self.assertEqual(type(testCL), list, 'The type of Classification.computeClassification(classifier) output is incorrect.')

    def testClassificationListLen(self):
        testCL = self.ClassificationTest.computeListIndividuals(self._classifier)
        self.assertEqual(len(testCL), 30, 'The Classification.computeClassification(classifier) resulting list contains the wrong number of elements.')

    def testClassificationEmptyList(self):
        testCL = self.ClassificationTest.computeListIndividuals(self._classifier)
        self.assertNotEqual(len(testCL), 0, 'The Classification.computeClassification(classifier) output list is empty.')

    def testClassificationTrueElement(self):
        testCL = self.ClassificationTest.computeListIndividuals(self._classifier)
        self.assertIn('http://wit.istc.cnr.it/geno/TARHBD-468', testCL, 'The tested element is not in the Classification.computeClassification(classifier) resulting list.')

    def testClassificationFalseElement(self):
        testCL = self.ClassificationTest.computeListIndividuals(self._classifier)
        self.assertNotIn(('no file'), testCL, 'The tested item is in the Classification.computeClassification(classifier) resulting list even if it shouldn''t.')

#VIII test case.
class ConfusionMatrixTestCase(unittest.TestCase):
    def testPresenceDirectory(self): #setUp and tearDown are not needed for this check
        self.assertTrue(os.path.exists("static"), 'The ''static'' folder has not been created.')

    def testPresenceMatrix(self): #setUp and tearDown are not needed for this check
        self.assertTrue(os.path.isfile('static/confusion-matrix.png'), 'The confusion matrix has not been created.')
    
        
#Initialize test suite
def suite():
    suite = unittest.TestSuite()
    suite.addTest(DataSetReaderTestCase('testQuery'))
    suite.addTest(DataSetReaderTestCase('testQueryTypeCheck'))
    suite.addTest(DataSetReaderTestCase('testQueryKeys'))
    suite.addTest(DataSetReaderTestCase('testQueryItems'))
    suite.addTest(DataSetReaderTestCase('testQueryKeysLen'))
    suite.addTest(DataSetReaderTestCase('testQueryItemsLen'))
    suite.addTest(DataSetReaderTestCase('testQueryTrueKey'))
    suite.addTest(DataSetReaderTestCase('testQueryFalseItem'))
    suite.addTest(StatisticsRecorderTestCase('testStatistics'))
    suite.addTest(StatisticsRecorderTestCase('testStatisticsTypeCheck'))
    suite.addTest(StatisticsRecorderTestCase('testStatisticsKeys'))
    suite.addTest(StatisticsRecorderTestCase('testStatisticsItems'))
    suite.addTest(StatisticsRecorderTestCase('testStatisticsKeysLen'))
    suite.addTest(StatisticsRecorderTestCase('testStatisticsItemsLen'))
    suite.addTest(StatisticsRecorderTestCase('testStatisticsTrueKey'))
    suite.addTest(StatisticsRecorderTestCase('testStatisticsFalseItem'))
    suite.addTest(StatisticsRecorderTestCase('testStatisticsGreaterValue'))
    suite.addTest(TrainingGraphTestCase('testTrainingGraph'))
    suite.addTest(TrainingGraphTestCase('testTrainingGraphList'))
    suite.addTest(TrainingGraphTestCase('testTrainingGraphListLen'))
    suite.addTest(TrainingGraphTestCase('testTrainingGraphItemPresent'))
    suite.addTest(TrainingGraphTestCase('testTrainingGraphListNotEmpty'))
    suite.addTest(TestGraphTestCase('testTestGraph'))
    suite.addTest(TestGraphTestCase('testTestGraphList'))
    suite.addTest(TestGraphTestCase('testTestGraphListLen'))
    suite.addTest(TestGraphTestCase('testTestGraphItemPresent'))
    suite.addTest(TestGraphTestCase('testTestGraphListNotEmpty'))
    suite.addTest(ValidationGraphTestCase('testValidationGraph'))
    suite.addTest(ValidationGraphTestCase('testValidationGraphListLen'))
    suite.addTest(ValidationGraphTestCase('testValidationGraphItemPresent'))
    suite.addTest(ValidationGraphTestCase('testValidationGraphListNotEmpty'))
    suite.addTest(MachineLearningTestCase('testML'))
    suite.addTest(MachineLearningTestCase('testMLTypeCheck'))
    suite.addTest(MachineLearningTestCase('testMLKeys'))
    suite.addTest(MachineLearningTestCase('testMLItems'))
    suite.addTest(MachineLearningTestCase('testMLKeysLen'))
    suite.addTest(MachineLearningTestCase('testMLItemsLen'))
    suite.addTest(MachineLearningTestCase('testMLTrueKey'))
    suite.addTest(MachineLearningTestCase('testMLFalseItem'))
    suite.addTest(ClassificationTestCase('testClassification'))
    suite.addTest(ClassificationTestCase('testClassificationTypeCheck'))
    suite.addTest(ClassificationTestCase('testClassificationListLen'))
    suite.addTest(ClassificationTestCase('testClassificationEmptyList'))
    suite.addTest(ClassificationTestCase('testClassificationTrueElement'))
    suite.addTest(ClassificationTestCase('testClassificationFalseElement'))
    suite.addTest(ConfusionMatrixTestCase('testPresenceDirectory'))
    suite.addTest(ConfusionMatrixTestCase('testPresenceMatrix'))
    return suite

#Initialize a runner
if __name__=='__main__':
    output_template.main()
    runner.run(suite())
    
###########################################################################
##  If some of the tests fail, the user can uncomment this region
##  and comment out the previous one to find out which is the test
##  that has failed! 
##if __name__=='__main__':
##    result = unittest.TextTestRunner(verbosity=3).run(suite())
###########################################################################

#Loaders are not present because we put both cases and suites in the same file.


