from io import StringIO
from SPARQLWrapper import SPARQLWrapper, JSON
from flask import Flask, render_template, request
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools, os

#We use the Gaussian Naive Bayes as classifier.
classifier = GaussianNB()

class DataSetReader:
    def query(sparql):
        #Set the SPARQL endpoint URL
        endpoint = SPARQLWrapper("http://wit.istc.cnr.it/geno/sparql")

        #This query allows to get the data representing the training set.
        endpoint.setQuery(sparql)
        endpoint.setReturnFormat(JSON)
        return endpoint.query().convert()
    
    def rdfjson2pandas(json_results):
        data = ""
        for result in json_results["results"]["bindings"]:
            data += result["split"]["value"] 
        
            for c in result["sequence"]["value"]:
                value = ord(c)
                data += "," + str(value)
            if "class" in result: 
                data += "," + result["class"]["value"] + "\n"
    
        #Generate a Pandas data frame from the in-memory CSV so that it can be used by scikit-learn and return it.
        return pd.read_csv(StringIO(data), sep=",", index_col=0)

class StatisticsRecorder: 
    def computeStats(): 
    #We retrieve the number of individuals from training graph.
        sparql = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT (COUNT(DISTINCT ?split) AS ?inds)
            FROM <http://wit.istc.cnr.it/geno/training>
            WHERE {
                ?split <http://wit.istc.cnr.it/geno/sequence> ?sequence
            }
        """
        results = DataSetReader.query(sparql)
        trainingGraphStats = results["results"]["bindings"][0]["inds"]["value"]
        
        #Then, we retrieve the number of individuals from test graph.
        sparql = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT (COUNT(DISTINCT ?split) AS ?inds)
            FROM <http://wit.istc.cnr.it/geno/test>
            WHERE {
                ?split <http://wit.istc.cnr.it/geno/sequence> ?sequence
            }
            """
        results = DataSetReader.query(sparql)
        testGraphStats = results["results"]["bindings"][0]["inds"]["value"]
        
        #Then, we retrieve the number of individuals from validation graph.
        sparql = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT (COUNT(DISTINCT ?split) AS ?inds)
            FROM <http://wit.istc.cnr.it/geno/validation>
            WHERE {
                ?split <http://wit.istc.cnr.it/geno/sequence> ?sequence
            }
            """
        results = DataSetReader.query(sparql)
        validationGraphStats = results["results"]["bindings"][0]["inds"]["value"]
        
        return {"trainingGraphStats": trainingGraphStats,
                "testGraphStats": testGraphStats,
                "validationGraphStats": validationGraphStats}

class MachineLearning:   
    def trainClassifier(classifier): 
        training = TrainingGraph.getTrainingGraph()
        test = TestGraph.getTestGraph()
        
        X_train, y_train = training.iloc[:, :-1], training.iloc[:, -1]
        X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

        #We train the classifier
        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)
        
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        
        cm = confusion_matrix(y_test, predictions)
        
        confusion_matrix_image = ConfusionMatrixCreator.computeSaveConfusionMatrix(cm, ["ie", "ei", "n"])
        
        return          {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion_matrix": confusion_matrix_image

                        }
    
class PerformanceAnalyzer(MachineLearning):
    def showMetrics():
        return MachineLearning.trainClassifier(classifier)
    
class ConfusionMatrixCreator:    
    def computeSaveConfusionMatrix(cm, classes):
        cmap=plt.cm.Blues
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        fileName = "static/confusion-matrix.png"
        
        #The line below saves the confution matrix on a PNG file named out.png.
        plt.savefig(fileName, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches='tight', pad_inches=0.3,
                    frameon=None)
        
        return fileName

class TrainingGraph(DataSetReader): 
    def getTrainingGraph():
        sparql = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?split ?sequence ?class ?gene
            FROM <http://wit.istc.cnr.it/geno/training>
            WHERE {
                ?split <http://wit.istc.cnr.it/geno/sequence> ?sequence ;
                    <http://wit.istc.cnr.it/geno/classification> ?class
            }
            """
        results=DataSetReader.query(sparql)
        
        return DataSetReader.rdfjson2pandas(results)

class TestGraph(DataSetReader):
    def getTestGraph():
        sparql = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?split ?sequence ?class ?gene
            FROM <http://wit.istc.cnr.it/geno/test>
            WHERE {
                ?split <http://wit.istc.cnr.it/geno/sequence> ?sequence ;
                    <http://wit.istc.cnr.it/geno/classification> ?class
            }
            """
        results = DataSetReader.query(sparql)
        
        return DataSetReader.rdfjson2pandas(results)

class ValidationGraph(DataSetReader):
    def getValidationGraph():
        sparql = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?split ?sequence ?class ?gene
            FROM <http://wit.istc.cnr.it/geno/validation>
            WHERE {
                ?split <http://wit.istc.cnr.it/geno/sequence> ?sequence
            }
            """
        results = DataSetReader.query(sparql)
        
        return DataSetReader.rdfjson2pandas(results)

class Classification:
    def computeListIndividuals(classifier): #computes the list of individuals available for classification
        sparql = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?split ?sequence ?class ?gene
        FROM <http://wit.istc.cnr.it/geno/validation>
        WHERE {
            ?split <http://wit.istc.cnr.it/geno/sequence> ?sequence
        }


        """
        results = DataSetReader.query(sparql)
        
        individuals = []
        for result in results["results"]["bindings"]:
            
            sequence = ""
            for c in result["sequence"]["value"]:
                value = ord(c)
                if len(sequence) > 0:
                    sequence += ","
                
                sequence += str(value)
                
            ind = result["split"]["value"]
            individuals.append(ind)
            
        return individuals

    def classificationPrediction():
        ind = request.form['individual']            
        sparql = """
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?split ?sequence ?class ?gene
                FROM <http://wit.istc.cnr.it/geno/validation>
                WHERE {
                    <""" + ind + """> <http://wit.istc.cnr.it/geno/sequence> ?sequence
                }
                """
        results = DataSetReader.query(sparql)
        sequence = []
        for result in results["results"]["bindings"]:
            for c in result["sequence"]["value"]:
                #We append the integer representationg of the sequence element.
                sequence.append(ord(c))
                
        #Here we compute the prediction
        predictions = classifier.predict([sequence])
        return ind, predictions

# The app is available on port 5000 on your Web browser, i.e. http://localhost:5000
class WebPage:
    app = Flask(__name__)
    @app.route("/")
    def homepage():
        return render_template('homepage.html')

    @app.route('/statistics', methods=['POST'])
    def statisticsPage():
        stats = StatisticsRecorder.computeStats()
        return render_template('statistics.html',stats=stats)

    @app.route('/performance',methods=['POST'])
    def performancePage():
        metrics = PerformanceAnalyzer.showMetrics()
        return render_template('performance.html',metrics=metrics)

    @app.route('/classification', methods=['POST']) 
    def classificationSelectionPage():
        individuals = Classification.computeListIndividuals(classifier)
        return render_template('classification.html', validationSet=individuals)   

    @app.route('/classify',methods=['POST'])
    def classificationPredictionPage():
        classification,predictions = Classification.classificationPrediction()
        return render_template('prediction.html', individual=classification, predictions=predictions)

    trainingSet = TrainingGraph.getTrainingGraph()
    X_train, y_train = trainingSet.iloc[:, :-1], trainingSet.iloc[:, -1]

    classifier.fit(X_train, y_train)
               
    if __name__ == '__main__':
        classifier = GaussianNB()
    
        # We create the static folder that we will use for saving the image of the confusion matrix.
        # The static folder is then used by the HTML template engine (i.e. Flask) as default target for images. 
        if not os.path.exists("static"):
            os.mkdir("static")

    if __name__ == '__main__':   
        app.run(debug=True)
