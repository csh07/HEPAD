from userPackage.Package_Encode import EncodeAllFeatures
from userPackage.LoadDataset import LoadDataset
import pandas as pd
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Predict import Predict

class HEPAD_Predict:
    def __init__(self, model_use, pathDict):
        self.model_use = model_use
        self.pathDict = pathDict
        self.modelNameList = ['rbfsvm', 'lightgbm', 'gbc']
        self.dataList = []
        self.predVectorDf = None
        self.probVectorDf = None
        self.predVectorListIndp = None
        self.probVectorListIndp = None
        if self.model_use == '1':
            self.featureNum = 250
            self.featureTypeDictJson = 'HemoPi_1_featureTypeDict.json'
            self.nmlzPkl = 'HemoPi_1_standardScaler.pkl'
            self.featureRankCsv = '../data/mlData/Hmp1/featureRank_hmp1.csv'
        elif self.model_use == '2':
            self.featureNum = 90
            self.featureTypeDictJson = 'HemoPi_2_featureTypeDict.json'
            self.nmlzPkl = 'HemoPi_2_standardScaler.pkl'
            self.featureRankCsv = '../data/mlData/Hmp2/featureRank_hmp2.csv'
        elif self.model_use == '3':
            self.featureNum = 130
            self.featureTypeDictJson = 'HemoPi_3_featureTypeDict.json'
            self.nmlzPkl = 'HemoPi_3_standardScaler.pkl'
            self.featureRankCsv = '../data/mlData/Hmp3/featureRank_hmp3.csv'
        elif self.model_use == '4':
            self.featureNum = 350
            self.featureTypeDictJson = 'HemoPi_m_featureTypeDict.json'
            self.nmlzPkl = 'HemoPi_m_standardScaler.pkl'
            self.featureRankCsv = '../data/mlData/Hmpm/featureRank_hmpm.csv'
        else:
            raise NameError('model_use should input 1 or 2 or 3 or 4, 1 = Hmp1, 2 = Hmp2, 3 = Hmp3, 4 = Hmpm')

    def loadData(self, inputDataList):
        ldObj = LoadDataset()
        for inputData in inputDataList:
            testSeqDict = ldObj.readFasta(inputData, minSeqLength=5)
            self.dataList.append(testSeqDict)

    def featureEncode(self):
        encodeObj = EncodeAllFeatures()
        encodeObj.dataEncodeSetup(loadJsonPath=f'{self.pathDict["paramPath"]}/{self.featureTypeDictJson}')
        encodeObj.dataEncodeOutput(dataList=self.dataList)
        testDf = encodeObj.dataNormalization(loadNmlzScalerPklPath=f'{self.pathDict["paramPath"]}/{self.nmlzPkl}')
        featureDf = pd.read_csv(self.featureRankCsv)
        featureList = featureDf['feature name'].to_list()
        if self.model_use == '1':
            testDf['MotifBitVec_KKG'] = [0] * 220
            testDf = testDf[featureList]
            testDf.to_csv(f'{self.pathDict["saveCsvPath"]}/test_F{self.featureNum}.csv')
        elif self.model_use == '2':
            testDf['y'] = [0] * 201
            testDf = testDf[featureList]
            testDf.to_csv(f'{self.pathDict["saveCsvPath"]}/test_F{self.featureNum}.csv')
        else:
            testDf = testDf[featureList]
            testDf.to_csv(f'{self.pathDict["saveCsvPath"]}/test_F{self.featureNum}.csv')

    def doPredict(self):
        pycObj = PycaretWrapper()
        modelList = pycObj.doLoadModel(path=self.pathDict['modelPath'], fileNameList=self.modelNameList)
        dataTestDf = pd.read_csv(f'{self.pathDict["saveCsvPath"]}/test_F{self.featureNum}.csv', index_col=[0])
        predObjIndp = Predict(dataX=dataTestDf, modelList=modelList)
        self.predVectorListIndp, self.probVectorListIndp = predObjIndp.doPredict()
        self.predVectorDf = pd.DataFrame(self.predVectorListIndp, index=self.modelNameList, columns=dataTestDf.index).T
        self.probVectorDf = pd.DataFrame(self.probVectorListIndp, index=self.modelNameList, columns=dataTestDf.index).T
        self.predVectorDf.to_csv(f'{self.pathDict["outputPath"]}/binary_vector.csv')
        self.probVectorDf.to_csv(f'{self.pathDict["outputPath"]}/probability_vector.csv')
